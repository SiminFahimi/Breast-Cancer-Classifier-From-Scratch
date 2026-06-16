# eval.py
from plots import PlotCurve, plot_curves
import numpy as np
import math

def evaluate(network, x_test, y_test, regularize=False, lambda_=None):
    activations, _ = network.forward_pass(x_test, network.weights)
    h = activations[-1]
    cost = network.binary_cross_entropy_loss(
        h, y_test, network.weights, regularize=regularize, lambda_=lambda_
    )
    val_predictions = network.predict_classes(x_test)
    acc = np.mean(val_predictions == y_test.ravel())
    return cost, acc, val_predictions


def cross_validation(x_train, y_train, fold_num, num_epochs, net_factory, lr, lambda_, seed=42):
    if lambda_ is None:
        lambda_ = 0.0001

    n = len(x_train)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    folds = np.array_split(indices, fold_num)

    fold_cost = []
    fold_acc = []

    for i in range(fold_num):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(fold_num) if j != i])

        x_fold_train = x_train[train_idx]
        y_fold_train = y_train[train_idx]

        x_val = x_train[val_idx]
        y_val = y_train[val_idx]

        net = net_factory()
        net.fit(x_fold_train, y_fold_train, num_epochs, lr, lambda_)

        cost, acc, _ = evaluate(net, x_val, y_val, regularize=False, lambda_=lambda_)

        fold_cost.append(cost)
        fold_acc.append(acc)

    return np.mean(fold_acc), np.mean(fold_cost)


def lr_effect(dataset, learning_rates, fold_num, num_epochs, net_factory, lambda_, plot=False, seed=42):
    x_train = dataset.X_train
    y_train = dataset.Y_train

    cv_costs = []
    best_lr = None
    best_cost = math.inf

    for lr in learning_rates:
        cv_acc, cv_cost = cross_validation(
            x_train, y_train, fold_num, num_epochs, net_factory, lr, lambda_, seed=seed
        )
        cv_costs.append(cv_cost)

        if not np.isnan(cv_cost) and cv_cost < best_cost:
            best_cost = cv_cost
            best_lr = lr

    if best_lr is None:
        best_lr = min(learning_rates)

    if plot:
        curves = PlotCurve(
            data=[(learning_rates, cv_costs)],
            x_label="Learning Rate",
            y_label="Cost",
            line_label=["CV Cost"],
            title="Learning Rates Comparison",
            save_address="results/lr_effect.png",
        )
        plot_curves(curves)
    return best_lr


def lambda_effect(dataset, fold_num, num_epochs, net_factory, lr, lambdas, plot=True, seed=42):
    x_train = dataset.X_train
    y_train = dataset.Y_train

    cv_costs = []
    train_costs = []

    for lambda_ in lambdas:
        _, cv_cost = cross_validation(
            x_train, y_train, fold_num, num_epochs, net_factory, lr, lambda_, seed=seed
        )
        cv_costs.append(cv_cost)

        net = net_factory()
        net.fit(x_train, y_train, num_epochs, lr, lambda_)
        train_cost, _, _ = evaluate(net, x_train, y_train, regularize=False, lambda_=lambda_)
        train_costs.append(train_cost)

    if plot:
        curves = PlotCurve(
            data=[
                (lambdas, cv_costs),
                (lambdas, train_costs),
            ],
            x_label="Lambda",
            y_label="Cost",
            line_label=["CV Cost", "Train Cost"],
            title="Lambda Effect",
            save_address="results/lambda_effect.png",
        )
        plot_curves(curves)


def cost_each_epoch(dataset, best_lr, num_epochs, net_factory, lambda_):
    x_train = dataset.X_train
    y_train = dataset.Y_train

    net = net_factory()
    costs_of_each_epoch = net.fit(x_train, y_train, num_epochs, best_lr, lambda_)

    epochs = range(1, num_epochs + 1)
    best_curve_plot = PlotCurve(
        data=[(epochs, costs_of_each_epoch)],
        x_label="Epoch",
        y_label="Cost",
        line_label=[f"lr={best_lr}"],
        title="Cost Curve on Best LR",
        save_address="results/loss.png",
    )
    plot_curves(best_curve_plot)

    return net, costs_of_each_epoch


def size_effect_on_cost(datas, fold_num, net_factory, num_epochs, lambda_, lr, repeats=5, seed=42):
    cv_costs = []
    train_costs = []
    counts = []

    for size, dataset in sorted(datas.items()):
        counts.append(size)

        cv_rep = []
        train_rep = []

        for r in range(repeats):
            _, cv_cost = cross_validation(
                dataset.X_train, dataset.Y_train, fold_num, num_epochs, net_factory, lr, lambda_, seed=seed + r
            )
            cv_rep.append(cv_cost)

            net = net_factory()
            net.fit(dataset.X_train, dataset.Y_train, num_epochs, lr, lambda_)
            train_cost, _, _ = evaluate(net, dataset.X_train, dataset.Y_train, regularize=False, lambda_=lambda_)
            train_rep.append(train_cost)

        cv_costs.append(np.mean(cv_rep))
        train_costs.append(np.mean(train_rep))

    plot_curves(
        PlotCurve(
            data=[(counts, cv_costs), (counts, train_costs)],
            x_label="Number of Samples",
            y_label="Cost",
            line_label=["CV Cost", "Train Cost"],
            title="Size Effect on Cost",
            save_address="results/size_effect_on_cost.png",
        )
    )


def size_effect_on_accuracy(datas, net_factory, num_epochs, lambda_, lr, repeats=5, seed=42):
    test_accs = []
    counts = []

    for size, dataset in sorted(datas.items()):
        counts.append(size)
        acc_rep = []

        for r in range(repeats):
            net = net_factory()
            net.fit(dataset.X_train, dataset.Y_train, num_epochs, lr, lambda_)
            _, acc, _ = evaluate(net, dataset.X_test, dataset.Y_test)
            acc_rep.append(acc)

        test_accs.append(np.mean(acc_rep))

    plot_curves(
        PlotCurve(
            data=[(counts, test_accs)],
            x_label="Samples",
            y_label="Test Accuracy",
            line_label=["Accuracy"],
            title="Size Effect on Accuracy",
            save_address="results/size_effect_on_accuracy.png",
        )
    )