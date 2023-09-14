#if !defined(__DSTRUCTS_CUH__)
#define __DSTRUCTS_CUH__

#include <device_launch_parameters.h>

template <class T>
class Stack
{
private:
    T st[15];    // ((2**5)-2)/2 refracted rays at most considering maxDepth of 5 as maximum
    int top;
    int N;
public:
    __device__
    Stack(const int& N) : N(N)
    {
        top = -1;
    }
    __device__
    void push(T var)
    {
        if (top < N-1)
            st[++top] = var;
    }
    __device__
    T pop() { return st[top--]; }
    __device__
    bool isEmpty() { return top < 0 ? true : false; }
};


#endif