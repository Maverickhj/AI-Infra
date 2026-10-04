export function groups(tp: number, dp: number, rank: number) {
  if (!Number.isInteger(tp) || ![1, 2].includes(tp))
    throw Error("非法 TP：此模型只验证 TP1/2；nkv=2、nq=4、FFN=12 必须可分");
  if (!Number.isInteger(dp) || ![1, 2].includes(dp))
    throw Error("非法 DP：此参考仅验证 DP1/2");
  const world = tp * dp;
  if (!Number.isInteger(rank) || rank < 0 || rank >= world)
    throw Error("rank 必须为 0 到 " + (world - 1) + " 的整数");
  const local = rank % tp,
    data = Math.floor(rank / tp);
  return {
    world,
    tpRank: local,
    dpRank: data,
    tpMembers: Array.from({ length: tp }, (_, i) => data * tp + i),
    dpMembers: Array.from({ length: dp }, (_, i) => local + i * tp),
  };
}
