# コード説明

1. random.rs
- 占い箇所をランダムで決めて、配置候補を推定(M=2のみ対応)
- [AHC030のseed0のバババッて決まるやつは結構簡単に作れるよという話](https://qiita.com/aplysia/items/c3f2111110ac5043710a)
- [累積分布関数、erf、正規化関連参考](https://bowwowforeach.hatenablog.com/entry/2023/08/24/205427?)

2. climbing_slow.rs
- 占いの相互情報量が大きくなるように占いマスを追加・削除する山登り
- 計算量が多いのでTLE(3秒以上)する
- [相互情報量を学んでもっとうまくAHC030を解こう！](https://qiita.com/aplysia/items/29a4fb4573fc1b8dec79)

3. climbing_fast.rs
- climbing_slowから以下を改善
    - 想定される計算(占いマスの数、油田数)を事前にしておく
    - log(x/y)=log(x)-log(y)であることを利用して、logの計算をループ内せずに事前計算し、ループ内では浮動小数点の引き算のみ
- [AHC030 Writer解](https://atcoder.jp/contests/ahc030/submissions/50443474)
- [AHCラジオ](https://www.youtube.com/watch?v=YvCYsiu-TQs&t=6s)

4. climbing_best.rs
- climbing_fastから以下を改善
    - 各マスの相互情報量を求めて、情報量が大きい順に追加・削除をする山登り
    - 情報量がないマス（配置候補において、油田数が同じマス）は山登りの追加・削除するマスから除外
    - 改善される限り、各マスの追加・削除を最大で3回なめる
    - 情報量うがないマスは、占い時にコスト削減のために占いマスに追加

# 動作確認

```bash
./run.sh [ファイル名] [入力ファイル] [出力ファイル]
# random.rsを動かす場合
./run.sh random in out
```