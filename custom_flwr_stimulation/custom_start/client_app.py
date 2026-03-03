# Importing libraries
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from custom_start.task import load_data, net, test_fn, train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    # load model architecture from task file, and change its device
    model = net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Extract ArrayRecord from Message and convert to PyTorch state_dict and load it into our model
    state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(state_dict)

    # take data from context for trainig
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id)

    # train_fn(model, ...) here model is pass by reference, so no need to take the model back
    train_loss = train_fn(model, trainloader, epochs=1, device=device)
    # Construct and return reply Message
    # Include the locally-trained model
    model_record = ArrayRecord(model.state_dict())
    # Include some statistics such as the training loss
    # We also want to include the number of examples used for training
    # so the strategy in the ServerApp can do FedAvg
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    # RecordDict are the main payload type in Messages
    # We insert both the ArrayRecord and the MetricRecord into it
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    model = net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load weights for evaluation
    state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(state_dict)

    partition_id = context.node_config["partition-id"]
    _, testloader = load_data(partition_id)

    loss, accuracy = test_fn(model, testloader, device=device)

    # Return metrics back to the server
    return msg.create_reply(content={"loss": loss, "accuracy": accuracy})
