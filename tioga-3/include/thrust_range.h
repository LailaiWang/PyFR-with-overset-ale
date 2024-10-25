#ifndef __THRUST_RANGE_H__
#define __THRUST_RANGE_H__
#include <thrust/iterator/counting_iterator.h>

namespace thrust {
  namespace views {
    struct iota {
      thrust::counting_iterator<size_t> _start;
      thrust::counting_iterator<size_t> _end;
      iota(size_t start, size_t end) :
        _start{thrust::counting_iterator<size_t>(start)}, 
        _end{thrust::counting_iterator<size_t>(end)} 
      {

      }

      auto begin()->decltype(_start) {
        return _start;
      }

      auto end()->decltype(_end) {
        return _end;
      }
    };
  }
}

#endif
