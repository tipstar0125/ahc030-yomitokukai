#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use itertools::Itertools;
use proconio::input_interactive;
use rand::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

fn main() {
    let time_limit = 3.0;
    let time_keeper = TimeKeeper::new(time_limit - 0.05);
    let start_time = time_keeper.get_time();
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(0);

    let input = read_input();
    let ITER = 1e5 as usize;
    let mut state = State::new(&input, &mut rng, ITER);
    let mutual_info = MutualInfo::new(&input);

    for t in 0..2 * input.n2 {
        if time_keeper.isTimeOver() {
            break;
        }

        // 最初は盤面をランダム生成しているのでスキップ
        if t > 0 {
            // ターンの最後で尤度は対数尤度になっている
            // 配置候補の集合をクリアして、再度pool内の配置候補を入れなおす
            // 削除された配置候補の中に正解があった場合に復活できるようにする処置
            state.set.clear();
            for board in state.pool.iter() {
                state.set.insert(board.hash, board.prob);
            }

            let mut mx_prob = state.pool[0].prob;
            let mut now_prob = mx_prob;
            let mut now_board = state.pool[0].clone();

            // 時間経過で回数を減らす
            let ITER = ((ITER as f64) * (time_limit - time_keeper.get_time()).min(1.0)) as usize;
            for _ in 0..ITER {
                let coin = rng.gen_range(0..10);
                let optional_board = {
                    if coin < 3 {
                        // 1つのポリオミノを上下左右いずれかに1マス移動する近傍
                        now_board.move_one_square(&input, &mut rng, &state)
                    } else if coin < 4 {
                        // 1つのポリオミノをランダムに移動する近傍
                        now_board.move_random(&input, &mut rng, &state)
                    } else {
                        // 2つのポリオミノをランダムにスワップする近傍
                        now_board.swap_random(&input, &mut rng, &state)
                    }
                };

                if let Some(next_board) = optional_board {
                    if !state.set.contains_key(&next_board.hash)
                        && next_board.prob - mx_prob >= -10.0
                    {
                        state.set.insert(next_board.hash, next_board.prob);
                        state.pool.push(next_board.clone());
                    }
                    if now_prob <= next_board.prob
                        || rng.gen_bool((next_board.prob - now_prob).exp())
                    {
                        now_prob = next_board.prob;
                        now_board = next_board;
                    }
                }
                mx_prob.setmax(now_prob);
            }
            state.ln_prob_to_prob();
            state.normalize();
            state.sort_by_prob();
        }

        eprintln!("Pool num: {}", state.pool.len());

        // 配置候補全てを使って相互情報量の山登りを行うと遅いので、上位に絞って近似計算する
        // 時間経過で絞る数を減らしていく
        let size = state
            .pool
            .len()
            .min((1e3 * (time_limit - time_keeper.get_time()).min(1.0)) as usize);
        // 占いマスを追加・削除する山登り
        let fortune_coords = mutual_info.climbing(&input, &state.pool[..size]);

        // 占い後、ベイズ推定で対数尤度計算
        state.infer(fortune_coords, &input);

        // 確率が低い盤面は最後にtruncateするので、ソートしておく
        state.sort_by_prob();

        // 最も確率の高い盤面が一定以上なら答える
        if state.answer(0.8, &input) {
            break;
        }

        // 確率上位のみ残す
        // 時間経過で絞る数を減らしていく
        state
            .pool
            .truncate((1e3 * (time_limit - time_keeper.get_time()).min(1.0)) as usize);
    }

    let elapsed_time = time_keeper.get_time() - start_time;
    eprintln!("Elapsed time: {elapsed_time}");
}

fn calc_likelihood(k: usize, eps: f64, cnt: usize, ret: usize) -> f64 {
    let mu = (k as f64 - cnt as f64) * eps + cnt as f64 * (1.0 - eps);
    let sigma = (k as f64 * eps * (1.0 - eps)).sqrt();
    probability_in_range(
        mu,
        sigma,
        if ret == 0 { -100.0 } else { ret as f64 - 0.5 },
        ret as f64 + 0.5,
    )
}

// 各計算において、確率もしくは尤度が1e-6をした回る場合は枝刈りして計算量を抑える
const EPS: f64 = 1e-6;

struct MutualInfo {
    probs: Vec<Vec<Vec<(f64, f64)>>>,
    probs_lower: Vec<Vec<usize>>,
}

impl MutualInfo {
    fn new(input: &Input) -> Self {
        let mut probs = mat![vec![]; input.n2 + 1; input.oil_total + 1];
        let mut probs_lower = mat![0; input.n2 + 1; input.oil_total + 1];
        // 占いマスの数として想定される数k: 1～N^2
        for k in 1..=input.n2 {
            // 占われたマスの油田数の合計としてありうる数cnt: 0～oil_total
            for cnt in 0..=input.oil_total {
                // 占い結果の平均値を求めて、それを中心に枝刈りされるまで計算をする
                let mu = (k as f64 - cnt as f64) * input.eps + cnt as f64 * (1.0 - input.eps);
                let center = mu.round() as usize;
                for r in (0..=center).rev() {
                    let prob = calc_likelihood(k, input.eps, cnt, r);
                    if prob < EPS {
                        // 後の計算で計算不要箇所をスキップするために、下限の枝刈り箇所を保存
                        probs_lower[k][cnt] = r + 1;
                        break;
                    }
                    // 後の計算でlog計算した値も用いるので、事前に計算しておく
                    probs[k][cnt].push((prob, prob.ln()));
                }
                probs[k][cnt].reverse();
                for r in center + 1.. {
                    let prob = calc_likelihood(k, input.eps, cnt, r);
                    if prob < EPS {
                        break;
                    }
                    probs[k][cnt].push((prob, prob.ln()));
                }
            }
        }
        MutualInfo { probs, probs_lower }
    }
    fn calc(&self, k: usize, pool: &[Board], cnt: &[usize]) -> f64 {
        // 候補盤面Bが正解である確率がP(B)、クエリの結果がretとなる確率がP(ret)のとき、
        // 相互情報量は Σ P(ret|B)P(B) log(P(ret|B)/P(ret))
        let mut ret_probs = vec![]; // P(ret)
        for (board, &c) in pool.iter().zip(cnt) {
            if board.prob < EPS {
                continue;
            }
            let lower = self.probs_lower[k][c];
            let length = lower + self.probs[k][c].len();
            // 枝刈り上限までresize
            if ret_probs.len() < length {
                ret_probs.resize(length, 0.0);
            }

            // 枝刈り下限から計算
            for ((prob, _), ret_prob) in self.probs[k][c].iter().zip(ret_probs[lower..].iter_mut())
            {
                *ret_prob += board.prob * prob;
            }
        }

        // ループの中でlogの計算をしたくないので、事前にlog計算をしておく
        for ret_prob in ret_probs.iter_mut() {
            *ret_prob = ret_prob.ln();
        }

        let mut info = 0.0;
        for (board, &c) in pool.iter().zip(cnt) {
            if board.prob < EPS {
                continue;
            }
            // 枝刈り下限から計算
            let lower = self.probs_lower[k][c];
            for ((prob, prob_ln), ret_prob_ln) in
                self.probs[k][c].iter().zip(ret_probs[lower..].iter())
            {
                info += prob * board.prob * (prob_ln - ret_prob_ln); // log(x/y)=log(x)-log(y)
            }
        }
        info * (k as f64).sqrt()
    }
    fn climbing(&self, input: &Input, pool: &[Board]) -> Vec<Coord> {
        // 全ての配置候補で埋蔵量が同じマスは占っても情報が得られないので除外
        let mut same = DynamicMap2d::new(vec![true; input.n2], input.n);
        for board in pool.iter().skip(1) {
            for i in 0..input.n {
                for j in 0..input.n {
                    let coord = Coord::new(i, j);
                    same[coord] = same[coord] && (pool[0].oil_cnt[coord] == board.oil_cnt[coord]);
                }
            }
        }
        // 各マスを、そのマス単体でクエリした時の相互情報量の降順にソート
        let info_each_sq = {
            let mut ret = vec![];
            for i in 0..input.n {
                for j in 0..input.n {
                    let coord = Coord::new(i, j);
                    if same[coord] {
                        continue;
                    }
                    let mut cnt = vec![0; pool.len()];
                    for (i, board) in pool.iter().enumerate() {
                        cnt[i] = board.oil_cnt[coord] as usize;
                    }
                    let info = self.calc(1, pool, &cnt);
                    ret.push((info, coord));
                }
            }
            ret.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            ret
        };

        let mut k = 0;
        let mut cnt = vec![0; pool.len()];
        let mut fortune_map = DynamicMap2d::new(vec![false; input.n2], input.n);
        let mut best_score = 0.0;

        for _ in 0..3 {
            let mut changed = false;
            for (_, coord) in info_each_sq.iter() {
                // 既に占いマスなら削除(-1)、占いマスでなければ追加(+1)
                let delta = if fortune_map[*coord] { !0 } else { 1 };
                fortune_map[*coord] = !fortune_map[*coord];
                // 油田数を差分更新
                for (i, board) in pool.iter().enumerate() {
                    cnt[i] += board.oil_cnt[*coord] as usize * delta;
                }
                k += delta;

                if best_score.setmax(self.calc(k, pool, &cnt)) {
                    changed = true;
                } else {
                    // 相互情報量が改善しなければ、元に戻す
                    let delta = if fortune_map[*coord] { !0 } else { 1 };
                    fortune_map[*coord] = !fortune_map[*coord];
                    for (i, board) in pool.iter().enumerate() {
                        cnt[i] += board.oil_cnt[*coord] as usize * delta;
                    }
                    k += delta;
                }
            }
            if !changed {
                break;
            }
        }

        let mut fortune_coords = vec![];
        for i in 0..input.n {
            for j in 0..input.n {
                let coord = Coord::new(i, j);
                // 情報量がないマスは、コスト削減のため占いマスに追加する
                if same[coord] || fortune_map[coord] {
                    fortune_coords.push(coord);
                }
            }
        }
        fortune_coords
    }
}

struct State {
    pool: Vec<Board>,
    hashes: Vec<DynamicMap2d<u64>>,
    set: FxHashMap<u64, f64>,
    likelihoods_memo: Vec<Vec<f64>>,
    fortune_coords_memo: Vec<Vec<Coord>>,
    query_cnt_diff_memo: Vec<Vec<DynamicMap2d<usize>>>,
}

impl State {
    // ランダムで盤面生成
    fn new(input: &Input, rng: &mut rand_pcg::Pcg64Mcg, iter: usize) -> Self {
        let mut hashes: Vec<DynamicMap2d<u64>> = vec![];
        for m in 0..input.m {
            if m > 0 && input.minos[m] == input.minos[m - 1] {
                hashes.push(hashes[m - 1].clone());
            }
            let mut mp = DynamicMap2d::new(vec![0; input.n2], input.n);
            for i in 0..input.n - input.minos[m].height + 1 {
                for j in 0..input.n - input.minos[m].width + 1 {
                    let coord = Coord::new(i, j);
                    mp[coord] = rng.gen::<u64>();
                }
            }
            hashes.push(mp);
        }

        let mut pool = vec![];
        let mut set = FxHashMap::default();
        for _ in 0..iter {
            let mut mino_pos_coords = vec![];
            let mut oil_cnt = DynamicMap2d::new(vec![0; input.n2], input.n);
            let mut hash = 0;
            for m in 0..input.m {
                let i = rng.gen_range(0..input.n - input.minos[m].height + 1);
                let j = rng.gen_range(0..input.n - input.minos[m].width + 1);
                let coord = Coord::new(i, j);
                hash ^= hashes[m][coord];
                mino_pos_coords.push(coord);
                for &coord_diff in input.minos[m].coords.iter() {
                    oil_cnt[coord + coord_diff] += 1;
                }
            }
            // 同じ盤面は追加しない
            if set.contains_key(&hash) {
                continue;
            }
            set.insert(hash, 0.0);
            pool.push(Board {
                prob: 0.0,
                oil_cnt,
                hash,
                mino_pos_coords,
                query_cnt: vec![],
            })
        }
        let L = pool.len() as f64;
        for board in pool.iter_mut() {
            board.prob = 1.0 / L;
        }
        State {
            pool,
            hashes,
            set,
            likelihoods_memo: vec![],
            fortune_coords_memo: vec![],
            query_cnt_diff_memo: vec![],
        }
    }
    fn infer(&mut self, fortune_coords: Vec<Coord>, input: &Input) {
        let k = fortune_coords.len();
        let ret = query(&fortune_coords);
        let mut likelihoods = vec![];

        // 再計算用にクエリで想定される油田数全ての計算をしておく
        for cnt in 0..=input.oil_total {
            let likelihood = calc_likelihood(k, input.eps, cnt, ret);
            likelihoods.push(likelihood.ln());
        }

        // query_cntは尤度再計算で毎回使用するので、メモをしておく
        for board in self.pool.iter_mut() {
            let mut cnt = 0;
            for &coord in fortune_coords.iter() {
                cnt += board.oil_cnt[coord] as usize;
            }
            board.query_cnt.push(cnt);
        }

        // boardを近傍に変更する際に、query_cntを差分更新するようの情報をメモしておく
        let fortune_coords_set: FxHashSet<Coord> = fortune_coords.clone().into_iter().collect();
        let mut query_cnt_diff = vec![];
        for m in 0..input.m {
            let mut mp = DynamicMap2d::new(vec![1000; input.n2], input.n);
            for i in 0..input.n - input.minos[m].height + 1 {
                for j in 0..input.n - input.minos[m].width + 1 {
                    let mut cnt = 0;
                    let coord = Coord::new(i, j);
                    for &coord_diff in &input.minos[m].coords {
                        if fortune_coords_set.contains(&(coord + coord_diff)) {
                            cnt += 1;
                        }
                    }
                    mp[coord] = cnt;
                }
            }
            query_cnt_diff.push(mp);
        }
        self.query_cnt_diff_memo.push(query_cnt_diff);

        // 再計算メモ
        self.likelihoods_memo.push(likelihoods);
        self.fortune_coords_memo.push(fortune_coords);

        // 都度、最初のクエリから計算
        // 対数尤度にして、値が小さくなりすぎないようにする
        self.calc_all_ln_prob();
    }
    fn calc_ln_prob(&self, board: &Board) -> f64 {
        // 初期の確率は全て等しいので、全ての盤面において同じ値(0.0)で初期化
        let mut ln_prob = 0.0;
        for (likelihoods, cnt) in self.likelihoods_memo.iter().zip(&board.query_cnt) {
            ln_prob += likelihoods[*cnt];
        }
        ln_prob
    }
    fn calc_query_cnt(&self, board: &Board) -> Vec<usize> {
        let mut query_cnt = vec![];
        for fortune_coords in self.fortune_coords_memo.iter() {
            let mut cnt = 0;
            for &coord in fortune_coords.iter() {
                cnt += board.oil_cnt[coord] as usize;
            }
            query_cnt.push(cnt);
        }
        query_cnt
    }
    fn calc_query_cnt_diff(&self, q: usize, m: usize, before: Coord, after: Coord) -> usize {
        self.query_cnt_diff_memo[q][m][after] - self.query_cnt_diff_memo[q][m][before]
    }
    fn calc_all_ln_prob(&mut self) {
        for i in 0..self.pool.len() {
            self.pool[i].prob = self.calc_ln_prob(&self.pool[i]);
        }
    }
    fn ln_prob_to_prob(&mut self) {
        let mx = self
            .pool
            .iter()
            .max_by(|&a, &b| a.prob.partial_cmp(&b.prob).unwrap())
            .unwrap()
            .prob;
        for board in self.pool.iter_mut() {
            board.prob = (board.prob - mx).exp();
        }
    }
    fn normalize(&mut self) {
        let prob_sum = self.pool.iter().map(|board| board.prob).sum::<f64>();
        for board in self.pool.iter_mut() {
            board.prob /= prob_sum;
        }
    }
    fn sort_by_prob(&mut self) {
        self.pool
            .sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());
    }
    fn answer(&mut self, prob_threshold: f64, input: &Input) -> bool {
        // 対数尤度のままなので、事後確率にして正規化
        let mx = self.pool[0].prob;
        let prob_sum = self
            .pool
            .iter()
            .map(|board| (board.prob - mx).exp())
            .sum::<f64>();
        let mx = 1.0 / prob_sum;
        if mx > prob_threshold {
            let mut ans_coords = vec![];
            for i in 0..input.n {
                for j in 0..input.n {
                    let coord = Coord::new(i, j);
                    if self.pool[0].oil_cnt[coord] > 0 {
                        ans_coords.push(coord);
                    }
                }
            }
            let ret = answer(&ans_coords);
            if ret == 1 {
                return true;
            }
        }
        false
    }
}

#[derive(Debug, Clone)]
struct Board {
    prob: f64,
    oil_cnt: DynamicMap2d<u8>,
    hash: u64,
    mino_pos_coords: Vec<Coord>,
    query_cnt: Vec<usize>,
}

impl Board {
    fn move_one_square(
        &self,
        input: &Input,
        rng: &mut rand_pcg::Pcg64Mcg,
        state: &State,
    ) -> Option<Board> {
        let m = rng.gen_range(0..input.m);
        let coord_diff = ADJ[rng.gen_range(0..4)];
        let now_coord = self.mino_pos_coords[m];
        let next_coord = now_coord + coord_diff;
        if next_coord.row > input.n - input.minos[m].height
            || next_coord.col > input.n - input.minos[m].width
        {
            return None;
        }
        let mut next_board = self.clone();
        next_board.mino_pos_coords[m] = next_coord;
        for &coord_diff in input.minos[m].coords.iter() {
            next_board.oil_cnt[now_coord + coord_diff] -= 1;
            next_board.oil_cnt[next_coord + coord_diff] += 1;
        }
        next_board.hash ^= state.hashes[m][now_coord] ^ state.hashes[m][next_coord];
        for (q, query_cnt) in next_board.query_cnt.iter_mut().enumerate() {
            *query_cnt += state.calc_query_cnt_diff(q, m, now_coord, next_coord);
        }
        next_board.prob = state.calc_ln_prob(&next_board);
        Some(next_board)
    }
    fn move_random(
        &self,
        input: &Input,
        rng: &mut rand_pcg::Pcg64Mcg,
        state: &State,
    ) -> Option<Board> {
        let m = rng.gen_range(0..input.m);
        let i = rng.gen_range(0..input.n - input.minos[m].height + 1);
        let j = rng.gen_range(0..input.n - input.minos[m].width + 1);
        let now_coord = self.mino_pos_coords[m];
        let next_coord = Coord::new(i, j);
        let mut next_board = self.clone();
        next_board.mino_pos_coords[m] = next_coord;
        for &coord_diff in input.minos[m].coords.iter() {
            next_board.oil_cnt[now_coord + coord_diff] -= 1;
            next_board.oil_cnt[next_coord + coord_diff] += 1;
        }
        next_board.hash ^= state.hashes[m][now_coord] ^ state.hashes[m][next_coord];
        for (q, query_cnt) in next_board.query_cnt.iter_mut().enumerate() {
            *query_cnt += state.calc_query_cnt_diff(q, m, now_coord, next_coord);
        }
        next_board.prob = state.calc_ln_prob(&next_board);
        Some(next_board)
    }
    fn swap_random(
        &self,
        input: &Input,
        rng: &mut rand_pcg::Pcg64Mcg,
        state: &State,
    ) -> Option<Board> {
        let p = rng.gen_range(0..input.m);
        let q = rng.gen_range(0..input.m);
        if p == q {
            return None;
        }
        let p_coord = self.mino_pos_coords[p];
        let q_coord = self.mino_pos_coords[q];
        if p_coord.row > input.n - input.minos[q].height
            || p_coord.col > input.n - input.minos[q].width
        {
            return None;
        }
        if q_coord.row > input.n - input.minos[p].height
            || q_coord.col > input.n - input.minos[p].width
        {
            return None;
        }
        let mut next_board = self.clone();
        next_board.mino_pos_coords[p] = q_coord;
        next_board.mino_pos_coords[q] = p_coord;
        for &coord_diff in input.minos[p].coords.iter() {
            next_board.oil_cnt[p_coord + coord_diff] -= 1;
            next_board.oil_cnt[q_coord + coord_diff] += 1;
        }
        for &coord_diff in input.minos[q].coords.iter() {
            next_board.oil_cnt[q_coord + coord_diff] -= 1;
            next_board.oil_cnt[p_coord + coord_diff] += 1;
        }
        next_board.hash ^= state.hashes[p][p_coord] ^ state.hashes[p][q_coord];
        next_board.hash ^= state.hashes[q][q_coord] ^ state.hashes[q][p_coord];
        for (i, query_cnt) in next_board.query_cnt.iter_mut().enumerate() {
            *query_cnt += state.calc_query_cnt_diff(i, p, p_coord, q_coord);
            *query_cnt += state.calc_query_cnt_diff(i, q, q_coord, p_coord);
        }
        next_board.prob = state.calc_ln_prob(&next_board);
        Some(next_board)
    }
}

#[derive(PartialEq)]
struct Mino {
    coords: Vec<CoordDiff>,
    height: usize,
    width: usize,
}

struct Input {
    n: usize,
    n2: usize,
    m: usize,
    eps: f64,
    minos: Vec<Mino>,
    oil_total: usize,
}

fn read_input() -> Input {
    input_interactive! {
        n: usize,
        m: usize,
        eps: f64,
        mut mino_coords2: [[(usize, usize)]; m]
    }
    // ハッシュ計算時、同じポリオミノを同一視するために事前にソート
    mino_coords2.sort();

    let mut minos = vec![];
    let mut oil_total = 0;
    for i in 0..m {
        oil_total += mino_coords2[i].len();
        let mut height = 0;
        let mut width = 0;
        let mut coords = vec![];
        for j in 0..mino_coords2[i].len() {
            let (row, col) = mino_coords2[i][j];
            height.setmax(row + 1);
            width.setmax(col + 1);
            let coord = CoordDiff::new(row as isize, col as isize);
            coords.push(coord);
        }
        minos.push(Mino {
            coords,
            height,
            width,
        })
    }
    Input {
        n,
        n2: n * n,
        m,
        eps,
        minos,
        oil_total,
    }
}

fn query(coords: &Vec<Coord>) -> usize {
    println!(
        "q {} {}",
        coords.len(),
        coords
            .iter()
            .map(|coord| format!("{} {}", coord.row, coord.col))
            .join(" ")
    );
    input_interactive! {ret: usize}
    ret
}

fn answer(coords: &Vec<Coord>) -> usize {
    println!(
        "a {} {}",
        coords.len(),
        coords
            .iter()
            .map(|coord| format!("{} {}", coord.row, coord.col))
            .join(" ")
    );
    input_interactive! {ret: usize}
    ret
}

// ここからライブラリ

pub trait SetMinMax {
    fn setmin(&mut self, v: Self) -> bool;
    fn setmax(&mut self, v: Self) -> bool;
}
impl<T> SetMinMax for T
where
    T: PartialOrd,
{
    fn setmin(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }
    fn setmax(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { vec![$($e),*] };
	($($e:expr,)*) => { vec![$($e),*] };
	($e:expr; $d:expr) => { vec![$e; $d] };
	($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
}

#[derive(Debug, Clone)]
struct TimeKeeper {
    start_time: std::time::Instant,
    time_threshold: f64,
}

impl TimeKeeper {
    fn new(time_threshold: f64) -> Self {
        TimeKeeper {
            start_time: std::time::Instant::now(),
            time_threshold,
        }
    }
    #[inline]
    fn isTimeOver(&self) -> bool {
        let elapsed_time = self.start_time.elapsed().as_nanos() as f64 * 1e-9;
        #[cfg(feature = "local")]
        {
            elapsed_time * 0.55 >= self.time_threshold
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time >= self.time_threshold
        }
    }
    #[inline]
    fn get_time(&self) -> f64 {
        let elapsed_time = self.start_time.elapsed().as_nanos() as f64 * 1e-9;
        #[cfg(feature = "local")]
        {
            elapsed_time * 0.55
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord {
    row: usize,
    col: usize,
}

impl Coord {
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
    pub fn in_map(&self, height: usize, width: usize) -> bool {
        self.row < height && self.col < width
    }
    pub fn to_index(&self, width: usize) -> CoordIndex {
        CoordIndex(self.row * width + self.col)
    }
}

impl std::ops::Add<CoordDiff> for Coord {
    type Output = Coord;
    fn add(self, rhs: CoordDiff) -> Self::Output {
        Coord::new(
            self.row.wrapping_add_signed(rhs.dr),
            self.col.wrapping_add_signed(rhs.dc),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoordDiff {
    dr: isize,
    dc: isize,
}

impl CoordDiff {
    pub const fn new(dr: isize, dc: isize) -> Self {
        Self { dr, dc }
    }
}

pub const ADJ: [CoordDiff; 4] = [
    CoordDiff::new(1, 0),
    CoordDiff::new(!0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(0, !0),
];

pub struct CoordIndex(pub usize);

impl CoordIndex {
    pub fn new(index: usize) -> Self {
        Self(index)
    }
    pub fn to_coord(&self, width: usize) -> Coord {
        Coord {
            row: self.0 / width,
            col: self.0 % width,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynamicMap2d<T> {
    pub size: usize,
    map: Vec<T>,
}

impl<T> DynamicMap2d<T> {
    pub fn new(map: Vec<T>, size: usize) -> Self {
        assert_eq!(size * size, map.len());
        Self { size, map }
    }
}

impl<T> std::ops::Index<Coord> for DynamicMap2d<T> {
    type Output = T;

    #[inline]
    fn index(&self, coordinate: Coord) -> &Self::Output {
        &self[coordinate.to_index(self.size)]
    }
}

impl<T> std::ops::IndexMut<Coord> for DynamicMap2d<T> {
    #[inline]
    fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
        let size = self.size;
        &mut self[coordinate.to_index(size)]
    }
}

impl<T> std::ops::Index<CoordIndex> for DynamicMap2d<T> {
    type Output = T;

    fn index(&self, index: CoordIndex) -> &Self::Output {
        unsafe { self.map.get_unchecked(index.0) }
    }
}

impl<T> std::ops::IndexMut<CoordIndex> for DynamicMap2d<T> {
    #[inline]
    fn index_mut(&mut self, index: CoordIndex) -> &mut Self::Output {
        unsafe { self.map.get_unchecked_mut(index.0) }
    }
}

#[allow(clippy::approx_constant)]
fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    0.5 * (1.0 + libm::erf((x - mean) / (std_dev * 1.41421356237)))
}

fn probability_in_range(mean: f64, std_dev: f64, a: f64, b: f64) -> f64 {
    if mean < a {
        return probability_in_range(mean, std_dev, 2.0 * mean - b, 2.0 * mean - a);
    }
    let p_a = normal_cdf(a, mean, std_dev);
    let p_b = normal_cdf(b, mean, std_dev);
    p_b - p_a
}
