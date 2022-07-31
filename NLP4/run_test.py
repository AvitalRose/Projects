import argparse
import torch
from torch import nn, optim
from snli_data import SnliData
from model import DistanceBasedModel


def test(data, model, loss_function, epoch_num,device):
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
    parser.add_argument('--device', default="cuda:0")
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

    criterion = nn.CrossEntropyLoss()



    with open("results.txt", "w") as f:
        f.write("results are:\n")

    # train
    for i in range(1):
        trained_model = DistanceBasedModel(args=args, embedding_dict=snli_data_embedding).to(args.device)
        trained_model.load_state_dict(torch.load(args.saved_model))
        train_loss, train_acc, train_size = test(data=train_data, model=trained_model, loss_function=criterion,
                                                 epoch_num=i, device=args.device)
        val_loss, val_acc, val_size = test(data=dev_data, model=trained_model, loss_function=criterion, epoch_num=i,
                                           device=args.device)
        test_loss, test_acc, test_size = test(data=test_data, model=trained_model, loss_function=criterion, epoch_num=i,
                                              device=args.device)
        print(f"train loss is: {train_loss} train_accuracy is: {train_acc} train size: {train_size}")
        print(f"dev loss is: {val_loss} dev accuracy is: {val_acc} dev size: {val_size}")
        print(f"test loss is: {test_loss} test accuracy is: {test_acc} test size: {test_size}")
        with open("results.txt", 'a') as f:
            f.write(f"train loss is: {train_loss} train_accuracy is: {train_acc} train size: {train_size}\n")
            f.write(f"dev loss is: {val_loss} dev accuracy is: {val_acc} dev size: {val_size}\n")
            f.write(f"test loss is: {test_loss} test accuracy is: {test_acc} test size: {test_size}\n")





