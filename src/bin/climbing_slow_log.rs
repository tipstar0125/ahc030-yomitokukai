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
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(12345);

    loop {
        // 占いマスを追加・削除する山登り
        let fortune_coords = climbing(&input, &pool, &mut rng);
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
            board.prob = board.prob.ln() + likelihood.ln();
        }

        // 対数尤度→事後確率→正規化
        let mx = pool
            .iter()
            .max_by(|&a, &b| a.prob.partial_cmp(&b.prob).unwrap())
            .unwrap()
            .prob;
        let mut prob_sum = 0.0;
        for board in pool.iter_mut() {
            board.prob = (board.prob - mx).exp();
            prob_sum += board.prob;
        }
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

fn climbing(input: &Input, pool: &[Board], rng: &mut rand_pcg::Pcg64Mcg) -> Vec<Coord> {
    // 初期値は全てのマスを占うとして山登りをスタート
    let mut fortune_map = DynamicMap2d::new(vec![true; input.n2], input.n);
    let mut best_score = 0.0;
    let mut k = input.n2;

    // 各候補盤面の占いマスにおける油田数を事前計算
    let mut cnt = vec![0; pool.len()];
    for (c, board) in cnt.iter_mut().zip(pool) {
        for i in 0..input.n {
            for j in 0..input.n {
                let coord = Coord::new(i, j);
                if fortune_map[coord] {
                    *c += board.oil_cnt[coord] as usize;
                }
            }
        }
    }

    for _ in 0..1000 {
        let i = rng.gen_range(0..input.n);
        let j = rng.gen_range(0..input.n);
        let coord = Coord::new(i, j);

        // 既に占いマスなら削除(-1)、占いマスでなければ追加(+1)
        let delta = if fortune_map[coord] { !0 } else { 1 };
        fortune_map[coord] = !fortune_map[coord];
        // 油田数を差分更新
        for (i, board) in pool.iter().enumerate() {
            cnt[i] += board.oil_cnt[coord] as usize * delta;
        }
        k += delta;

        let score = calc_mutual_information(k, input, pool, &cnt);
        if score > best_score {
            best_score = score;
        } else {
            // 相互情報量が改善しなければ、元に戻す
            let delta = if fortune_map[coord] { !0 } else { 1 };
            fortune_map[coord] = !fortune_map[coord];
            for (i, board) in pool.iter().enumerate() {
                cnt[i] += board.oil_cnt[coord] as usize * delta;
            }
            k += delta;
        }
    }
    let mut fortune_coords = vec![];
    for i in 0..input.n {
        for j in 0..input.n {
            let coord = Coord::new(i, j);
            if fortune_map[coord] {
                fortune_coords.push(coord);
            }
        }
    }
    fortune_coords
}

// 各計算において、確率もしくは尤度が1e-6をした回る場合は枝刈りして計算量を抑える
const EPS: f64 = 1e-6;

// 候補盤面Bが正解である確率がP(B)、クエリの結果がretとなる確率がP(ret)のとき、
// 相互情報量は Σ P(ret|B)P(B) log(P(ret|B)/P(ret))
fn calc_mutual_information(k: usize, input: &Input, pool: &[Board], cnt: &[usize]) -> f64 {
    let mut ret_probs = vec![]; // P(ret)
    for (board, &c) in pool.iter().zip(cnt) {
        if board.prob < EPS {
            continue;
        }
        // 占い結果の平均値を求めて、それを中心に枝刈りされるまで計算をする
        let mu = (k as f64 - c as f64) * input.eps + c as f64 * (1.0 - input.eps);
        let center = mu.round() as usize;
        for r in (0..=center).rev() {
            let prob = calc_likelihood(k, input.eps, c, r);
            if prob < EPS {
                break;
            }
            // 長さが足りない場合は、resize
            if ret_probs.len() < r + 1 {
                ret_probs.resize(r + 1, 0.0);
            }
            ret_probs[r] += board.prob * prob;
        }
        for r in center + 1.. {
            let prob = calc_likelihood(k, input.eps, c, r); // P(ret|B)
            if prob < EPS {
                break;
            }
            if ret_probs.len() < r + 1 {
                ret_probs.resize(r + 1, 0.0);
            }
            ret_probs[r] += board.prob * prob;
        }
    }

    let mut info = 0.0;
    for (ret, &ret_prob) in ret_probs.iter().enumerate() {
        if ret_prob < EPS {
            continue;
        }
        for (board, &c) in pool.iter().zip(cnt) {
            if board.prob < EPS {
                continue;
            }
            let likelihood = calc_likelihood(k, input.eps, c, ret);
            if likelihood < EPS {
                continue;
            }
            info += likelihood * board.prob * (likelihood / ret_prob).log2();
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
