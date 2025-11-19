### This is a combined script to run different cases sequentially
### Test 1: the open-loop results of the Oracle in Discovery Learning
import case1_main
case1_main.Testing()
### Test 2: the closed-loop results of the Learner in the first-round unsupervised active learning, i.e., the uncertainty results:
import main_22_to_15
main_22_to_15.Testing()
### Test 3: the closed-loop results of the Oracle in Discovery Learning (at the end of active learning)
import case2_main
case2_main.Testing()
### Test 4: the closed-loop results of the Learner in the second-round supervised active learning, i.e., the final predictions of the Learner:
import main_26_to_11
main_26_to_11.Testing()