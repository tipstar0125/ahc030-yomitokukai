from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from common.settings import TEST_PATH

test_case_num = 100
thread_num = 4
score_list = [0] * test_case_num
cost_list = [0] * test_case_num
tle1_list = []
tle2_list = []


def worker(n):
    path = Path(TEST_PATH)
    filename = str(n).zfill(4)
    cmd = (
        f"cargo run -r --manifest-path tools/Cargo.toml --bin tester cargo run -r --features local --bin a < tools/in/{filename}.txt > tools/out/{filename}.txt"
    )
    proc = subprocess.Popen(cmd, shell=True, cwd=path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_list = proc.communicate()

    score = 0
    elapsed_time = 0
    for stderr in stderr_list:
        out_list = stderr.decode().split("\n")
        for out in out_list:
            if "score" in out.lower():
                score = int(out.split("=")[-1])
                if n < test_case_num:
                    score_list[n] = score

            if "elapsed" in out.lower():
                t = int(float(out.split(":")[-1]) * 1000)
                elapsed_time = t
                if t >= 2950:
                    tle1_list.append((n, t))
                if t >= 3000:
                    tle2_list.append((n, t))

    print(f"{filename} score: {score}, elapsed: {elapsed_time}ms")
    if elapsed_time >= 3000:
        print("TLE")


def main():
    if TEST_PATH is None:
        return

    with open("score.txt", "r+") as f:
        f.truncate(0)

    n = 0

    while n < test_case_num:
        t1 = threading.Thread(target=worker, args=(n,))
        t2 = threading.Thread(target=worker, args=(n + 1,))
        t3 = threading.Thread(target=worker, args=(n + 2,))
        t4 = threading.Thread(target=worker, args=(n + 3,))
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        n += thread_num

    score_sum = sum(score_list)
    score_average = score_sum / len(score_list)
    cost_sum = sum(cost_list)
    cost_average = cost_sum / len(cost_list)
    print(score_average)
    print(score_sum)
    print(cost_average)
    print(tle1_list)
    print(tle2_list)

    for score in score_list:
        with open("score.txt", "a") as f:
            f.write(str(score) + "\n")


if __name__ == "__main__":
    main()
