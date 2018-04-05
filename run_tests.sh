#!/bin/bash

SCRIPT=/home/vault/capm/sn0515/PhD/Th_U-Wire/script_test.py

echo '========================================'
echo 'run Th228 MC @S5 Check'
echo '========================================'
(python ${SCRIPT} -events 50000 --mc -multi MS -position S5 ${@:1})

echo '========================================'
echo 'run Gamma MC Check'
echo '========================================'
(python ${SCRIPT} -events 50000 --gamma -multi MS ${@:1})

echo '========================================'
echo 'run Th228 Data @S5 Check'
echo '========================================'
#(python ${SCRIPT} -events 50000 --data -multi MS -position S5 ${@:1})
(python ${SCRIPT} -events 50000 --rotate -multi MS -position S5 ${@:1})

echo '========================================'
echo 'run Th228 Data @S2 Check'
echo '========================================'
#(python ${SCRIPT} -events 50000 --rotate -multi MS -position S2 ${@:1})
(python ${SCRIPT} -events 50000 --data -multi MS -position S2 ${@:1})

echo '========================================'
echo 'run Th228 Data @S8 Check'
echo '========================================'
#(python ${SCRIPT} -events 24000   --rotate -multi MS -position S8 ${@:1})
(python ${SCRIPT} -events 24000 --data -multi MS -position S8 ${@:1})


echo '==================== Checks finished ===================='
