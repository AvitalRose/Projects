import argparse
import torch
from torch import nn, optim
import pickle
from snli_data import SnliData
from model import DistanceBasedModel
import matplotlib.pyplot as plt


def train(data, model, optimizer, loss_function, epoch_num, device):
    model.train()
    loss, acc, size = 0, 0, 0
    for idx, batch in enumerate(data):
        premise, premise_lengths, hypothesis, hypothesis_lengths, y = batch
        inputs = (premise, premise_lengths, hypothesis, hypothesis_lengths)

        output = model(inputs)
        optimizer.zero_grad()
        y = y.type(torch.LongTensor)
        y = y.to(torch.device(device))
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        _, prediction = output.max(dim=1)
        acc += (prediction == y).sum().float()
        size += len(prediction)

    return model, loss, acc/size, size


def validate(data, model, loss_function, epoch_num, device):
    model.eval()
    acc, loss, size = 0, 0, 0
    for idx, batch in enumerate(data):
        premise, premise_lengths, hypothesis, hypothesis_lengths, y = batch
        inputs = (premise, premise_lengths, hypothesis, hypothesis_lengths)
        output = model(inputs)

        y = y.type(torch.LongTensor)
        y = y.to(torch.device(device))
        batch_loss = loss_function(output, y)
        loss += batch_loss.item()

        _, prediction = output.max(dim=1)
        acc += (prediction == y).sum().float()
        size += len(prediction)
    acc /= size
    acc = acc.cpu().item()
    return loss, acc, size

def test(data, model, loss_function, epoch_num, device):
    model.eval()
    acc, loss, size = 0, 0, 0
    for idx, batch in enumerate(data):
        premise, premise_lengths, hypothesis, hypothesis_lengths, y = batch
        inputs = (premise, premise_lengths, hypothesis, hypothesis_lengths)
        output = model(inputs)

        y = y.type(torch.LongTensor)
        y = y.to(torch.device(device))
        batch_loss = loss_function(output, y)
        loss += batch_loss.item()

        _, prediction = output.max(dim=1)
        acc += (prediction == y).sum().float()
        size += len(prediction)
    acc /= size
    acc = acc.cpu().item()

    return loss, acc, size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-train', default="snli_1.0/snli_1.0_train.txt")
    parser.add_argument('--file-dev', default="snli_1.0/snli_1.0_dev.txt")
    parser.add_argument('--file-test', default="snli_1.0/snli_1.0_test.txt")
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--embedding-dim', default=300, type=int)
    parser.add_argument('--d_ff', default=300 * 4, type=int)
    parser.add_argument('--max-premise', default=83, type=int)
    parser.add_argument('--max-hypothesis', default=83, type=int)
    parser.add_argument('--shuffle', default=False)
    parser.add_argument('--num-heads', default=5, type=int)
    parser.add_argument('--alpha', default=1.5, type=float)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--targets-size', default=3, type=int)
    parser.add_argument('--relu-layer-dim', default=300, type=int)
    parser.add_argument('--saved-model', default="distance_based_saved_model")
    args = parser.parse_args()

    # get data
    snli_data = SnliData(args=args)
    train_data, dev_data, test_data = snli_data.get_data_loaders()
    snli_data_embedding = snli_data.index_embedding

    #
    # # # used for faster development
    # # with open("train_data.pkl", 'wb') as file:
    # #     # A new file will be created
    # #     pickle.dump(train_data, file)
    # # with open("test_data.pkl", "wb") as file:
    # #     pickle.dump(test_data, file)
    # # with open("dev_data.pkl", "wb") as file:
    # #     pickle.dump(dev_data, file)
    # # with open("snli_data_embedding.pkl", 'wb') as file:
    # #     # A new file will be created
    # #     pickle.dump(snli_data.index_embedding, file)
    #
    # with open('train_data.pkl', 'rb') as file:
    #     # Call load method to deserialize
    #     train_data = pickle.load(file)
    # with open('test_data.pkl', 'rb') as file:
    #     # Call load method to deserialize
    #     test_data = pickle.load(file)
    # with open('dev_data.pkl', 'rb') as file:
    #     dev_data = pickle.load(file)
    # with open('snli_data_embedding.pkl', 'rb') as file:
    #     # Call load method to deserialize
    #     snli_data_embedding = pickle.load(file)

    # initialize model
    # distance_based_model = DistanceBasedModel(args, train_data, snli_data.index_embedding)  # restore later
    distance_based_model = DistanceBasedModel(args=args, embedding_dict=snli_data_embedding)
    distance_based_model.to(torch.device(args.device))

    # initialize optimizer
    # get only learnable parameters (don't want to train embedding)
    parameters = filter(lambda p: p.requires_grad, distance_based_model.parameters())
    model_optimizer = optim.Adam(parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    with open("final_results", "w") as f:
        f.write("results are:\n")

    # train
    train_losses, valid_losses, test_losses = [], [], []
    train_accuracies, valid_accuracies, test_accuracies = [], [], []
    for i in range(20):
        final_model, train_loss, train_acc, train_size = train(data=train_data, model=distance_based_model,
                                                               optimizer=model_optimizer, loss_function=criterion,
                                                               epoch_num=i, device=args.device)
        val_loss, val_acc, val_size = validate(data=dev_data, model=distance_based_model, loss_function=criterion,
                                               epoch_num=i, device=args.device)
        torch.save(final_model.state_dict(), args.saved_model)
        trained_model = DistanceBasedModel(args=args, embedding_dict=snli_data_embedding).to(args.device)
        trained_model.load_state_dict(torch.load(args.saved_model))
        test_loss, test_acc, test_size = test(data=test_data, model=trained_model, loss_function=criterion, epoch_num=i,
                                              device=args.device)
        print(f"train loss is: {train_loss} train_accuracy is: {train_acc} train size: {train_size}")
        print(f"dev loss is: {val_loss} dev accuracy is: {val_acc} dev size: {val_size}")
        print(f"test loss is: {test_loss} test accuracy is: {test_acc} test size: {test_size}")
        with open("final_results", 'a') as f:
            f.write(f"train loss is: {train_loss} train_accuracy is: {train_acc} train size: {train_size}\n")
            f.write(f"dev loss is: {val_loss} dev accuracy is: {val_acc} dev size: {val_size}\n")
            f.write(f"test loss is: {test_loss} test accuracy is: {test_acc} test size: {test_size}\n")
        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(val_acc)
        test_accuracies.append(test_acc)

    # plot results
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="validation")
    plt.plot(test_losses, label="test")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Loss for SNLI dataset\nDistance-based Self-Attention Network")
    plt.legend()
    plt.show()

    plt.plot(train_accuracies, label="train")
    plt.plot(valid_accuracies, label="validation")
    plt.plot(test_accuracies, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("Accuracy for SNLI dataset\nDistance-based Self-Attention Network")
    plt.legend()
    plt.show()





