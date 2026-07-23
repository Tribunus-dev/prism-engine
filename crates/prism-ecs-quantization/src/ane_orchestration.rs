use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AneTileDispatch { pub row: usize, pub col: usize, pub rows: usize, pub cols: usize, pub depth: usize }
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AneTiledDispatchPlan { pub rows: usize, pub cols: usize, pub depth: usize, pub dispatches: Vec<AneTileDispatch> }

pub fn plan_tiled_int8(rows: usize, cols: usize, depth: usize, tile_rows: usize, tile_cols: usize) -> Result<AneTiledDispatchPlan, String> {
    if rows == 0 || cols == 0 || depth == 0 || tile_rows == 0 || tile_cols == 0 { return Err("tile dimensions must be non-zero".into()); }
    let mut dispatches=Vec::new();
    for row in (0..rows).step_by(tile_rows) { for col in (0..cols).step_by(tile_cols) { dispatches.push(AneTileDispatch { row, col, rows:(rows-row).min(tile_rows), cols:(cols-col).min(tile_cols), depth }); } }
    Ok(AneTiledDispatchPlan { rows, cols, depth, dispatches })
}

pub fn execute_tiled_int8<F>(plan: &AneTiledDispatchPlan, activation: &[i8], weights: &[i8], mut execute: F) -> Result<Vec<i32>, String>
where F: FnMut((usize,usize), &[i8], &[i8]) -> Result<Vec<i32>, String> {
    if activation.len() < plan.rows*plan.depth || weights.len() < plan.depth*plan.cols { return Err("input buffers are smaller than the dispatch plan".into()); }
    let mut output=vec![0i32; plan.rows*plan.cols];
    for tile in &plan.dispatches { let mut a=Vec::with_capacity(tile.rows*plan.depth); for r in tile.row..tile.row+tile.rows { a.extend_from_slice(&activation[r*plan.depth..(r+1)*plan.depth]); } let mut w=Vec::with_capacity(plan.depth*tile.cols); for k in 0..plan.depth { w.extend_from_slice(&weights[k*plan.cols+tile.col..k*plan.cols+tile.col+tile.cols]); } let values=execute((tile.rows,tile.cols),&a,&w)?; if values.len()!=tile.rows*tile.cols{return Err("tile executor returned an invalid result length".into())}; for r in 0..tile.rows { output[(tile.row+r)*plan.cols+tile.col..(tile.row+r)*plan.cols+tile.col+tile.cols].copy_from_slice(&values[r*tile.cols..(r+1)*tile.cols]); } }
    Ok(output)
}
