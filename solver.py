import torch.nn.functional as F
import torch
import numpy as np
from torch import optim
from torch.autograd import Variable
from functools import reduce

import shl_processing
from dataloader import shl_loaders, SHL
from models import Encoder, Classifier


def accuracy(predicted_labels, true_labels):
    _, pred = torch.max(predicted_labels, 1)

    correct = np.squeeze(pred.eq(true_labels.data.view_as(pred)))
    return correct.float().mean() * 100


class Solver:
    def __init__(self):
        self.train_lr = 1e-5
        self.num_classes = 9
        self.clf_target = Classifier().cuda()
        self.clf2 = Classifier().cuda()
        self.clf1 = Classifier().cuda()
        self.encoder = Encoder().cuda()
        self.pretrain_lr = 1e-5
        self.weights_coef = 1e-3

    def to_var(self, x):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=False).float()

    def loss(self, predictions, y_true, weights_coef=None):
        """
        :param predictions: list of prediction tensors
        """
        assert len(predictions[0].shape) == 2 and len(y_true.shape) == 1, (predictions.shape, y_true.shape)
        losses = [F.cross_entropy(y_hat, y_true) for y_hat in predictions]
        loss = sum(losses)

        # """
        # We add the term |W1^T W2| to the cost function, where W1, W2 denote fully connected layers’
        # weights of F1 and F2 which are first applied to the feature F(xi)
        # """
        if weights_coef:
            lw = torch.matmul(solver.clf1.fc1.weight, solver.clf2.fc1.weight.T).abs().sum().mean()
            loss += weights_coef * lw

        return loss  # + lw

    def pretrain(self, source_loader, target_val_loader, pretrain_epochs=1):
        """
        TODO: add select max acc logic in code
        """
        source_iter = iter(source_loader)
        source_per_epoch = len(source_iter)
        print("source_per_epoch:", source_per_epoch)

        # pretrain
        log_pre = 250
        lr = self.pretrain_lr
        pretrain_iters = source_per_epoch * pretrain_epochs
        params = reduce(lambda a, b: a + b,
                        map(lambda i: list(i.parameters()),
                            [self.encoder, self.clf1, self.clf2, self.clf_target]))
        pretrain_optimizer = optim.Adam(params, lr)

        for step in range(pretrain_iters + 1):
            # ============ Initialization ============#
            # refresh
            if (step + 1) % source_per_epoch == 0:
                source_iter = iter(source_loader)
            # load the data
            source, s_labels = next(source_iter)
            source, s_labels = self.to_var(source), self.to_var(s_labels).long().squeeze()

            # ============ Training ============ #
            pretrain_optimizer.zero_grad()
            # forward
            features = self.encoder(source)
            y1_hat = self.clf1(features)
            y2_hat = self.clf2(features)
            y_target_hat = self.clf_target(features)

            # loss
            loss_source_class = self.loss([y1_hat, y2_hat, y_target_hat], s_labels, weights_coef=self.weights_coef)

            # one step
            loss_source_class.backward()
            pretrain_optimizer.step()
            pretrain_optimizer.zero_grad()
            # TODO: make this each step and on log_pre step just average and print previous results
            # ============ Validation ============ #
            if (step + 1) % log_pre == 0:
                with torch.no_grad():
                    source_val_features = self.encoder(source)
                    c_source1 = self.clf1(source_val_features)
                    c_source2 = self.clf2(source_val_features)
                    c_target = self.clf_target(source_val_features)
                    print("Train data (source) scores:")
                    print("Step %d | Source clf1=%.2f, clf2=%.2f | Target clf_t=%.2f" \
                          % (step,
                             accuracy(c_source1, s_labels),
                             accuracy(c_source2, s_labels),
                             accuracy(c_target, s_labels))
                          )
                    acc = self.eval(target_val_loader, self.clf_target)
                    print("Val data acc=%.2f" % acc)
                    print()

    def pseudo_labeling(self, loader, pool_size=4000, threshold=0.9):
        """
        When C1, C2 denote the class which has the maximum predicted probability for
        y1, y2, we assign a pseudo-label to xk if the following two
        conditions are satisfied. First, we require C1 = C2 to give
        pseudo-labels, which means two different classifiers agree
        with the prediction. The second requirement is that the
        maximizing probability of y1 or y2 exceeds the threshold
        parameter, which we set as 0.9 or 0.95 in the experiment.

        :return:
        """
        pool = []  # x, y_pseudo
        for x, _ in loader:
            batch_size = x.shape[0]
            x = self.to_var(x)
            ys1 = F.softmax(self.clf1(self.encoder(x)))
            ys2 = F.softmax(self.clf2(self.encoder(x)))
            # _, pseudo_labels = torch.max(pseudo_labels, 1)
            for i in range(batch_size):
                y1 = ys1[i]
                y2 = ys2[i]
                val1, idx1 = torch.max(y1, 0)
                val2, idx2 = torch.max(y2, 0)
                if idx1 == idx2 and max(val1, val2) >= threshold:
                    pool.append((x[i].cpu(), idx1.cpu().item()))
                if len(pool) >= pool_size:
                    return pool
        return pool

    def train(self, source_loader, source_val_loader, target_loader, target_val_loader, epochs):
        """
        :param epochs: target epochs the training will be done
        """

        # pretrain
        log_pre = 30
        lr = self.train_lr

        params1 = reduce(lambda a, b: a + b,
                         map(lambda i: list(i.parameters()),
                             [self.encoder, self.clf1, self.clf2]))
        params2 = list(self.encoder.parameters()) + list(self.clf_target.parameters())
        optimizer1 = optim.Adam(params1, lr)
        optimizer2 = optim.Adam(params2, lr)

        for epoch in range(epochs):
            source_iter = iter(source_loader)
            target_iter = iter(target_loader)

            source_per_epoch = len(source_iter)
            target_per_epoch = len(target_iter)
            if epoch == 0:
                print("source_per_epoch, target_per_epoch:", source_per_epoch, target_per_epoch)

            # Fill candidates
            target_candidates = self.pseudo_labeling(target_loader)
            print("Target candidates len:", len(target_candidates))
            if len(target_candidates) == 0:
                target_candidates = self.pseudo_labeling(target_loader, threshold=0.0)
                print("Target candidates len:", len(target_candidates))
            target_candidates_loader = self.wrap_to_loader(target_candidates)
            for step, (target, t_labels) in enumerate(target_candidates_loader):
                if (step + 1) % source_per_epoch == 0:
                    source_iter = iter(source_loader)

                source, s_labels = next(source_iter)
                target, t_labels = self.to_var(target), self.to_var(t_labels).long().squeeze()
                source, s_labels = self.to_var(source), self.to_var(s_labels).long().squeeze()

                # ============ Train F, F1, F2  ============ #
                optimizer1.zero_grad()
                # Source data
                # forward
                features = self.encoder(source)
                y1_hat = self.clf1(features)
                y2_hat = self.clf2(features)
                # loss
                loss_source_class = self.loss([y1_hat, y2_hat], s_labels, weights_coef=self.weights_coef)

                # Target data
                # forward
                features = self.encoder(target)
                y1_hat = self.clf1(features)
                y2_hat = self.clf2(features)
                # loss
                loss_target_class = self.loss([y1_hat, y2_hat], t_labels, weights_coef=self.weights_coef)
                # one step
                (loss_source_class + loss_target_class).backward()
                optimizer1.step()
                optimizer1.zero_grad()

                # ============ Train F, Ft  ============ #
                optimizer2.zero_grad()
                # Target data
                # forward
                y_target_hat = self.clf_target(self.encoder(target))
                # loss
                loss_target_class = self.loss([y_target_hat], t_labels)
                # one step
                loss_target_class.backward()
                optimizer2.step()
                optimizer2.zero_grad()

                # ============ Validation ============ #
                if (step + 1) % log_pre == 0:
                    acc = self.eval(target_val_loader, self.clf_target)
                    print("Step %d | Val data target classifier acc=%.2f" % (step, acc))
                    acc1 = self.eval(source_val_loader, self.clf1)
                    print("        | Val data source classifier1 acc=%.2f" % acc1)
                    acc2 = self.eval(source_val_loader, self.clf2)
                    print("        | Val data source classifier2 acc=%.2f" % acc2)
                    print()

    def save_models(self):
        torch.save(self.encoder, 'encoder.pth')
        torch.save(self.clf1, 'clf1.pth')
        torch.save(self.clf2, 'clf2.pth')
        torch.save(self.clf_target, 'clf_target.pth')

    def load_models(self):
        self.encoder = torch.load('encoder.pth')
        self.clf1 = torch.load('clf1.pth')
        self.clf2 = torch.load('clf2.pth')
        self.clf_target = torch.load('clf_target.pth')

    def eval(self, loader, classifier):
        """
        Evaluate encoder + passed classifier
        """
        # for x, y_true in loader:
        #     y_hat = classifier(self.encoder)
        #     acc = accuracy(y_hat, y_true)

        class_correct = [0] * self.num_classes
        class_total = [0.] * self.num_classes
        classes = shl_processing.coarse_label_mapping
        self.encoder.eval()
        classifier.eval()

        for x, y_true in loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            x, y_true = self.to_var(x), self.to_var(y_true).long().squeeze()

            y_hat = classifier(self.encoder(x))
            _, pred = torch.max(y_hat, 1)
            correct = np.squeeze(pred.eq(y_true.data.view_as(pred)))
            # calculate test accuracy for each object class
            for i in range(len(y_true.data)):
                label = y_true.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        for i in range(self.num_classes):
            if class_total[i] > 0:
                print('\tTest Accuracy of %10s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('\tTest Accuracy of %10s: N/A (no training examples)' % (classes[i]))

        self.encoder.train()
        classifier.train()

        return 100. * np.sum(class_correct) / np.sum(class_total)

    def wrap_to_loader(self, target_candidates):
        """
        :param target_candidates: [(x,y_pseudo)]
        :return:
        """
        assert len(target_candidates) > 0
        tmp = target_candidates  # CondomDataset(target_candidates)
        return torch.utils.data.DataLoader(dataset=tmp,
                                           batch_size=32,
                                           shuffle=True,
                                           num_workers=0)


# class CondomDataset(torch.utils.data.Dataset):
#     def __init__(self, data):
#         self.data = data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, item):
#         return self.data[item]


if __name__ == '__main__':
    source_train_loader, source_val_loader, target_train_loader, target_val_loader = shl_loaders()

    solver = Solver()
    solver.pretrain(source_loader=source_train_loader,
                    target_val_loader=target_val_loader,
                    pretrain_epochs=10)
    solver.save_models()

    solver.load_models()
    print("Accuracy before UDA:")
    print(solver.eval(target_val_loader, solver.clf_target))
    solver.train(source_train_loader,
                 source_val_loader,
                 target_train_loader,
                 target_val_loader, epochs=10)
    print("Final accuracy:")
    print(solver.eval(target_val_loader, solver.clf_target))
