import wandb
import time

wandb.init(
    project="SURA",
    name="hello-world-test",
    config={"test-run": True, "dummy_learning_rate": 0.001}
)

print("Transmitting dummy data to W&B")

for epoch in range(10):
    fake_loss = 5.0 / (epoch + 1)
    fake_acc = epoch * 10.0

    wandb.log(
        {
            "epoch": epoch,
            "train_loss": fake_loss,
            "train_acc": fake_acc
        }
    )

    time.sleep(0.5)

wandb.finish()
print("Run complete!")