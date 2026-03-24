import argparse, os, re
from torch.utils.tensorboard import SummaryWriter

EPOCH_LINE = re.compile(
    r"Epoch:\s*(\d+),\s*Steps:\s*(\d+)\s*\|\s*Train Loss:\s*([0-9.eE+-]+)\s*"
    r"Vali Loss:\s*([0-9.eE+-]+)\s*Test Loss:\s*([0-9.eE+-]+)"
)
EARLY_COUNTER = re.compile(r"EarlyStopping counter:\s*(\d+)\s*out of\s*(\d+)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    w = SummaryWriter(args.out)

    last_counter = None
    last_patience = None
    n = 0

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m2 = EARLY_COUNTER.search(line)
            if m2:
                last_counter = int(m2.group(1))
                last_patience = int(m2.group(2))

            m = EPOCH_LINE.search(line)
            if not m:
                continue

            epoch = int(m.group(1))
            steps = int(m.group(2))
            train_loss = float(m.group(3))
            vali_loss = float(m.group(4))
            test_loss = float(m.group(5))

            w.add_scalar("loss/train", train_loss, epoch)
            w.add_scalar("loss/vali", vali_loss, epoch)
            w.add_scalar("loss/test", test_loss, epoch)
            w.add_scalar("steps_per_epoch", steps, epoch)
            if last_counter is not None and last_patience is not None:
                w.add_scalar("earlystop/counter", last_counter, epoch)
                w.add_scalar("earlystop/patience", last_patience, epoch)

            n += 1

    w.flush()
    w.close()
    print(f"[OK] parsed {n} epochs -> {args.out}")

if __name__ == "__main__":
    main()
