#[derive(Copy, Clone)]
    pub enum AllignedPair<'a, T>{
        HasBoth(&'a T, &'a T),
        HasLeft(&'a T),
        HasRight(&'a T),
    }

    pub fn allign<'a, T,I,R>(v1: &Vec<T>, v2: &Vec<T>, get_id: &'a dyn Fn(&T) -> I, map: &'a mut dyn FnMut(AllignedPair<T>) -> R) -> Vec<R> where I: std::cmp::PartialOrd{
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