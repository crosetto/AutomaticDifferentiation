#include <utility>
#include <iostream>
template<typename T1, typename T2>
struct plus;
template<typename T1, typename T2>
struct times;

template<typename T, typename U>
constexpr auto operator+(T l, U r){ return plus(l,r);}
template<typename T, typename U>
constexpr auto operator*(T l, U r){ return times(l,r);}

template<typename T1, typename T2>
struct plus{
    constexpr plus(T1 t1_, T2 t2_):t1{t1_},t2{t2_}{}
    T1 t1;
    T2 t2;
    template<typename T>
    constexpr auto operator()(T&& t) const {return t1(std::forward<T>(t))+t2(std::forward<T>(t));}
};

template<typename T>
struct is_plus : std::false_type{};

template<typename T1, typename T2>
struct is_plus<plus<T1, T2>> : std::true_type{};

template<typename T1, typename T2>
struct times{
    constexpr times(T1 t1_, T2 t2_):t1{t1_},t2{t2_}{}
    T1 t1;
    T2 t2;
    template<typename T>
    constexpr auto operator()(T&& t) const {return t1(std::forward<T>(t))*t2(std::forward<T>(t));}
};

template<typename T>
struct is_times : std::false_type{};

template<typename T1, typename T2>
struct is_times<times<T1, T2>> : std::true_type{};

template <typename T, typename U>
using same = std::is_same<typename std::decay<T>::type,typename std::decay<U>::type>;

int main(){

    constexpr auto x = [](auto t){ return t ;};
    constexpr auto c = [](auto t){ return  [t](auto){return t;};};

    constexpr auto zero = [](auto t){ return 0 ;};
    constexpr auto one = [](auto t){ return 1 ;};

    constexpr auto recursion = [one, zero, x](auto t, auto self){
        using t_t = decltype(t);
        using x_t = decltype(x);
        if constexpr (same<t_t,x_t>::value)
	    return one;
        else if constexpr( is_plus<typename std::decay<decltype(t)>::type>::value)
            return plus{self(t.t1, self), self(t.t2, self)};
        else if constexpr( is_times<typename std::decay<decltype(t)>::type>::value)
            return plus{times{self(t.t1, self), t.t2}, times{t.t1, self(t.t2, self)}};
        else
            return zero;
    };

    constexpr auto D = [recursion](auto t){return recursion(t, recursion);};

    auto constexpr ex = D(x*x*c(4.));
    auto constexpr val=ex(5.);
    std::cout<<val<<"\n";
}

