use rustc_hash::FxBuildHasher;
use indexmap::IndexMap;

#[derive(Copy, Clone)]
pub enum AllignedPair<'a, T>{
    HasBoth(&'a T, &'a T),
    HasLeft(&'a T),
    HasRight(&'a T),
}

pub fn allign<'a, T, I, R, F, M>(v1: &Vec<T>, v2: &Vec<T>, get_id: &'a F, map: &'a mut M) -> Vec<R>
where 
    I: std::cmp::PartialOrd,
    F: Fn(&T) -> I,
    M: FnMut(AllignedPair<T>) -> R
{
    let n1 = v1.len();
    let n2 = v2.len();
    let n_res = std::cmp::max(n1,n2);
    let mut i1 = 0;
    let mut i2 = 0;
    let mut res = Vec::with_capacity(n_res);

    while i1 < n1 || i2 < n2 {
        if i1 < n1 {
            let x1 = &v1[i1];
            let id1 = get_id(x1);
            if i2 < n2 {
                //still processing v1 and v2
                let x2 = &v2[i2];
                let id2 = get_id(x2);
                if id1 == id2 {
                    let pair = AllignedPair::HasBoth(x1, x2);
                    res.push(map(pair));
                    i1 += 1;
                    i2 += 1;
                } else if id1 < id2 {
                    let pair = AllignedPair::HasLeft(x1);
                    res.push(map(pair));
                    i1 += 1;
                } else {
                    let pair = AllignedPair::HasRight(x2);
                    res.push(map(pair));
                    i2 += 1;
                }
            } else {
                //still processing v1 but finished with v2
                let pair = AllignedPair::HasLeft(x1);
                res.push(map(pair));
                i1 += 1;
            }
        } else {
            //finished processing ar1 but still busy with ar2
            let x2 = &v2[i2];
            let pair = AllignedPair::HasRight(x2);
            res.push(map(pair));
            i2 += 1;
        }
    }
    res
}

//todo: add tests
pub enum AllignedTuplePair<'a, K, V> {
    HasBoth((&'a K, &'a V), (&'a K, &'a V)),
    HasLeft((&'a K, &'a V)),
    HasRight((&'a K, &'a V)),
}

type FxIndexMap<K, V> = IndexMap<K, V, FxBuildHasher>;

pub fn allign_indexmap_map<'a, K, V, I, F, M>(m1: &FxIndexMap<K,V>, m2: &FxIndexMap<K,V>, get_id: &'a F, map: &'a mut M) -> FxIndexMap<K,V> 
where 
    I: std::cmp::PartialOrd,
    K: std::cmp::Eq + std::hash::Hash,
    F: Fn((&K, &V)) -> I,
    M: FnMut(AllignedTuplePair<K, V>) -> Option<(K, V)>,
{
    let n1 = m1.len();
    let n2 = m2.len();
    let n_res = std::cmp::max(n1,n2);
    let mut i1 = 0;
    let mut i2 = 0;
    let mut res: FxIndexMap<K,V> = IndexMap::with_capacity_and_hasher(n_res, FxBuildHasher::default());
    while i1 < n1 || i2 < n2 {
        if i1 < n1 {
            let x1 = m1.get_index(i1).unwrap();
            let id1 = get_id(x1);
            if i2 < n2 {
                //still processing v1 and v2
                let x2 = m2.get_index(i2).unwrap();
                let id2 = get_id(x2);
                if id1 == id2 {
                    let pair = AllignedTuplePair::HasBoth(x1, x2);
                    if let Some((k,v)) = map(pair) {
                        res.insert(k,v);
                    }
                    i1 += 1;
                    i2 += 1;
                } else if id1 < id2 {
                    let pair = AllignedTuplePair::HasLeft(x1);
                    if let Some((k,v)) = map(pair) {
                        res.insert(k,v);
                    }
                    i1 += 1;
                } else {
                    let pair = AllignedTuplePair::HasRight(x2);
                    if let Some((k,v)) = map(pair) {
                        res.insert(k,v);
                    }
                    i2 += 1;
                }
            } else {
                //still processing v1 but finished with v2
                let pair = AllignedTuplePair::HasLeft(x1);
                if let Some((k,v)) = map(pair) {
                    res.insert(k,v);
                }
                i1 += 1;
            }
        } else {
            //finished processing ar1 but still busy with ar2
            let x2 = m2.get_index(i2).unwrap();
            let pair = AllignedTuplePair::HasRight(x2);
            if let Some((k,v)) = map(pair) {
                res.insert(k,v);
            }
            i2 += 1;
        }
    }
    res
}

pub fn allign_indexmap_iter<'a, K, V, I, F, M>(m1: &FxIndexMap<K,V>, m2: &FxIndexMap<K,V>, get_id: &'a F, map: &'a mut M) 
where 
    I: std::cmp::PartialOrd,
    K: std::cmp::Eq + std::hash::Hash,
    F: Fn((&K,&V)) -> I,
    M: FnMut(AllignedTuplePair<K,V>) -> ()
{
    let n1 = m1.len();
    let n2 = m2.len();
    let mut i1 = 0;
    let mut i2 = 0;

    while i1 < n1 || i2 < n2 {
        if i1 < n1 {
            let x1 = m1.get_index(i1).unwrap();
            let id1 = get_id(x1);
            if i2 < n2 {
                //still processing v1 and v2
                let x2 = m2.get_index(i2).unwrap();
                let id2 = get_id(x2);
                if id1 == id2 {
                    let pair = AllignedTuplePair::HasBoth(x1, x2);
                    map(pair);
                    i1 += 1;
                    i2 += 1;
                } else if id1 < id2 {
                    let pair = AllignedTuplePair::HasLeft(x1);
                    map(pair);
                    i1 += 1;
                } else {
                    let pair = AllignedTuplePair::HasRight(x2);
                    map(pair);
                    i2 += 1;
                }
            } else {
                //still processing v1 but finished with v2
                let pair = AllignedTuplePair::HasLeft(x1);
                map(pair);
                i1 += 1;
            }
        } else {
            //finished processing ar1 but still busy with ar2
            let x2 = m2.get_index(i2).unwrap();
            let pair = AllignedTuplePair::HasRight(x2);
            map(pair);
            i2 += 1;
        }
    }
}