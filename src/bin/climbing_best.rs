#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use std::collections::{HashSet, VecDeque};

use itertools::Itertools;
use proconio::input_interactive;
use rand::prelude::*;

fn main() {
    let time_keeper = TimeKeeper::new(2.9);
    let start_time = time_keeper.get_time();

    let input = read_input();
    let mut pool = make_board(&input); // 盤面候補の生成
    let precalc = precalc_ret_probs(&input);

    loop {
        // 占いマスを追加・削除する山登り
        let fortune_coords = climbing(&input, &pool, &precalc);
        let k = fortune_coords.len();

        // ベイズ推定
        let ret = query(&fortune_coords);
        for board in pool.iter_mut() {
            let mut cnt = 0;
            for &coord in fortune_coords.iter() {
                cnt += board.oil_cnt[coord] as usize;
            }
            // ある候補盤面Bにおいて、retとなる尤度P(ret|B)を求める
            let likelihood = calc_likelihood(k, input.eps, cnt, ret);
            // P(B|ret)
            board.prob *= likelihood;
        }

        // 正規化
        let prob_sum = pool.iter().map(|board| board.prob).sum::<f64>();
        for board in pool.iter_mut() {
            board.prob /= prob_sum;
        }

        // 最も確率が高い盤面を求める
        let (idx, mx) = pool
            .iter()
            .map(|board| board.prob)
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // 80%以上なら答える
        if mx > 0.8 {
            let mut ans_coords = vec![];
            for i in 0..input.n {
                for j in 0..input.n {
                    let coord = Coord::new(i, j);
                    if pool[idx].oil_cnt[coord] > 0 {
                        ans_coords.push(coord);
                    }
                }
            }
            let ret = answer(&ans_coords);
            if ret == 1 {
                break;
            }
            // 誤りの場合、その盤面の確率を0にして、占いを繰り返す
            pool[idx].prob = 0.0;
        }
    }

    let elapsed_time = time_keeper.get_time() - start_time;
    eprintln!("Elapsed time: {elapsed_time}");
}

// BFSで全ての盤面を生成
fn make_board(input: &Input) -> Vec<Board> {
    let mut pool = vec![];
    let mut Q = VecDeque::new();
    Q.push_back((0, DynamicMap2d::new(vec![0; input.n2], input.n)));

    while let Some((c, oil_cnt)) = Q.pop_front() {
        if c == input.m {
            pool.push(Board { prob: 0.0, oil_cnt });
            continue;
        }
        for i in 0..input.n - input.minos[c].height + 1 {
            for j in 0..input.n - input.minos[c].width + 1 {
                let mut next_oil_cnt = oil_cnt.clone();
                let coord = Coord::new(i, j);
                for &diff in input.minos[c].coords.iter() {
                    next_oil_cnt[coord + diff] += 1;
                }
                Q.push_back((c + 1, next_oil_cnt));
            }
        }
    }
    let L = pool.len() as f64;
    for board in pool.iter_mut() {
        board.prob = 1.0 / L;
    }
    pool
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

fn climbing(input: &Input, pool: &[Board], precalc: &PreCalc) -> Vec<Coord> {
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
                let info = calc_mutual_information(1, pool, &cnt, precalc);
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

            if best_score.setmax(calc_mutual_information(k, pool, &cnt, precalc)) {
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

struct PreCalc {
    probs: Vec<Vec<Vec<(f64, f64)>>>,
    probs_lower: Vec<Vec<usize>>,
}

// 各計算において、確率もしくは尤度が1e-6をした回る場合は枝刈りして計算量を抑える
const EPS: f64 = 1e-6;

fn precalc_ret_probs(input: &Input) -> PreCalc {
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
    PreCalc { probs, probs_lower }
}

// 候補盤面Bが正解である確率がP(B)、クエリの結果がretとなる確率がP(ret)のとき、
// 相互情報量は Σ P(ret|B)P(B) log(P(ret|B)/P(ret))
fn calc_mutual_information(k: usize, pool: &[Board], cnt: &[usize], precalc: &PreCalc) -> f64 {
    let mut ret_probs = vec![]; // P(ret)
    for (board, &c) in pool.iter().zip(cnt) {
        if board.prob < EPS {
            continue;
        }
        let lower = precalc.probs_lower[k][c];
        let length = lower + precalc.probs[k][c].len();
        // 枝刈り上限までresize
        if ret_probs.len() < length {
            ret_probs.resize(length, 0.0);
        }

        // 枝刈り下限から計算
        for ((prob, _), ret_prob) in precalc.probs[k][c]
            .iter()
            .zip(ret_probs[lower..].iter_mut())
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
        let lower = precalc.probs_lower[k][c];
        for ((prob, prob_ln), ret_prob_ln) in
            precalc.probs[k][c].iter().zip(ret_probs[lower..].iter())
        {
            info += prob * board.prob * (prob_ln - ret_prob_ln); // log(x/y)=log(x)-log(y)
        }
    }
    info * (k as f64).sqrt()
}

struct Board {
    prob: f64,
    oil_cnt: DynamicMap2d<u8>,
}

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
        mino_coords2: [[(usize, usize)]; m]
    }
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

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone)]
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
