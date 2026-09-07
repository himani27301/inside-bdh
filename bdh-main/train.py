# Copyright Pathway Technology, Inc.

import os
from contextlib import nullcontext

import bdh
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# DEVICE SETUP
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use float32 on CPU
# Use bfloat16/float16 only when CUDA is available
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
    if torch.cuda.is_available()
    else "float32"
)

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)

scaler = torch.amp.GradScaler(
    device=device.type,
    enabled=(dtype == "float16")
)

torch.manual_seed(1337)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"Using device: {device} with dtype {dtype}")


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BDH_CONFIG = bdh.BDHConfig()

# Reduced values just to TEST whether BDH runs on your laptop
BLOCK_SIZE = 64
BATCH_SIZE = 2
MAX_ITERS = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 1


input_file_path = os.path.join(
    os.path.dirname(__file__),
    "input.txt"
)


# --------------------------------------------------
# FETCH TINY SHAKESPEARE DATASET
# --------------------------------------------------

def fetch_data():

    if not os.path.exists(input_file_path):

        print("Downloading Tiny Shakespeare dataset...")

        data_url = (
            "https://raw.githubusercontent.com/"
            "karpathy/char-rnn/master/data/"
            "tinyshakespeare/input.txt"
        )

        response = requests.get(data_url)

        with open(
            input_file_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(response.text)

        print("Dataset downloaded.")


# --------------------------------------------------
# CREATE TRAINING BATCH
# --------------------------------------------------

def get_batch(split):

    data = np.memmap(
        input_file_path,
        dtype=np.uint8,
        mode="r"
    )

    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)):]

    ix = torch.randint(
        len(data) - BLOCK_SIZE,
        (BATCH_SIZE,)
    )

    x = torch.stack(
        [
            torch.from_numpy(
                data[i:i + BLOCK_SIZE]
                .astype(np.int64)
            )
            for i in ix
        ]
    )

    y = torch.stack(
        [
            torch.from_numpy(
                data[
                    i + 1:
                    i + 1 + BLOCK_SIZE
                ].astype(np.int64)
            )
            for i in ix
        ]
    )

    if torch.cuda.is_available():

        x = x.pin_memory().to(
            device,
            non_blocking=True
        )

        y = y.pin_memory().to(
            device,
            non_blocking=True
        )

    else:

        x = x.to(device)
        y = y.to(device)

    return x, y


# --------------------------------------------------
# MAIN TRAINING
# --------------------------------------------------

if __name__ == "__main__":

    fetch_data()

    print("Creating BDH model...")

    model = bdh.BDH(BDH_CONFIG).to(device)

    # IMPORTANT:
    # torch.compile is disabled because your Windows CPU
    # currently does not have the MSVC "cl" compiler.
    #
    # model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    x, y = get_batch("train")

    print("\nStarting BDH test training...\n")

    loss_acc = 0.0
    loss_steps = 0

    for step in range(MAX_ITERS):

        with ctx:
            logits, loss = model(x, y)

        loss_acc += loss.item()
        loss_steps += 1

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        x, y = get_batch("train")

        if step % LOG_FREQ == 0:

            average_loss = (
                loss_acc / loss_steps
            )

            print(
                f"Step: {step}/{MAX_ITERS} "
                f"loss: {average_loss:.4f}"
            )

            loss_acc = 0.0
            loss_steps = 0


    # --------------------------------------------------
    # GENERATE SAMPLE
    # --------------------------------------------------
    torch.save(model.state_dict(), "bdh_test.pt")
    print("Model saved as bdh_test.pt")
    print("\nTraining done.")
    print("Generating sample...\n")

    model.eval()

    prompt = torch.tensor(
        bytearray(
            "To be or ",
            "utf-8"
        ),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():

        ret = model.generate(
            prompt,
            max_new_tokens=100,
            top_k=3
        )

    ret_decoded = bytes(
        ret
        .to(torch.uint8)
        .to("cpu")
        .squeeze(0)
    ).decode(
        errors="backslashreplace"
    )

    print("Generated text:")
    print("-------------------------")
    print(ret_decoded)
    print("-------------------------")