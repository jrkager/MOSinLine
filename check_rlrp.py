import main as M
from r101_segment_instance import construct_r101_instance
inst = construct_r101_instance(k=10, i=1, _main_module=M)
r = M.solve_rlrp(inst, "logs/check_rlrp.txt")
print("仓库大小:", r.depot_sizes)
print("店的分配:", r.customer_depot_assignment)