TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CPP=g++
ARGS=""
OS=$(uname)
if [ "$OS" = "Darwin" ]; then
  CPP=clang++
  ARGS="-undefined dynamic_lookup"
fi

$CPP -std=c++11 $ARGS -shared shuffle_op.cc -o shuffle_op.so -fPIC -I $TF_INC -O2
