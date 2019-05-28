#include <thrust/random/linear_congruential_engine.h>
#include <iostream>
int main(void)
{
  // create a minstd_rand object, which is an instance of linear_congruential_engine
  thrust::minstd_rand rng1;
  // output some random values to cout
  std::cout << rng1() << std::endl;
  std::cout << rng1() << std::endl;
  std::cout << rng1() << std::endl;
  // a random value is printed
  // create a new minstd_rand from a seed
  thrust::minstd_rand rng2(13);
  // discard some random values
  rng2.discard(13);
  // stream the object to an iostream
  std::cout << rng2 << std::endl;
  // rng2's current state is printed
  // print the minimum and maximum values that minstd_rand can produce
  std::cout << thrust::minstd_rand::min << std::endl;
  std::cout << thrust::minstd_rand::max << std::endl;
  // the range of minstd_rand is printed
  // save the state of rng2 to a different object
  thrust::minstd_rand rng3 = rng2;
  // compare rng2 and rng3
  std::cout << (rng2 == rng3) << std::endl;
  // 1 is printed
  // re-seed rng2 with a different seed
  rng2.seed(7);
  // compare rng2 and rng3
  std::cout << (rng2 == rng3) << std::endl;
  // 0 is printed
  return 0;
}