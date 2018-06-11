# An idiom for Automatic Differentiation (AD) in C++17

This example comes as a follow up from an example of generic programming, which we showed during a C++ course at CSCS a couple of 
years ago (see https://www.youtube.com/watch?v=cC9MtflQ_nI&t=2915s towards the end, and sorry for the poor quality). In that occasion
the goal was to show that using C++14 and constexpr we could write expression templates and automatic differentiation idioms with
a little effort and an effective syntax. The goal now is to show how this idiom can evolve using C++17 constexpr lambdas, further 
reducing the coding effort. 

DISCLAIMER: the purpose of this repo is to present a proof of concept, not a production-ready library, so all protections are skipped.

To recap, we start from the goal to achieve:
* Implementing univariate polynomials of the form
```C++
  constexpr auto expr = x*x*x+x*x+x
```
  in which "x" is the independent variable. the "x"s are generic: might be real numbers, matrices, 
  vectors, functions, ...
* Being able to lazily evaluate the expression
```C++
  expr(5.); expr(3u); 
```
* Being able to compute the result of the expression as a compile time constant
```C++
  constexpr res = expr(5.); 
```
* Being able to compute the symbolic arbitrary order derivative of the expressions
```C++
constexpr auto expr_dd = D(D(expr));
```

We will show how we can use constexpr lambdas and constexpr if to reduce simpliyf the code, and we show an extension of the grammar, including multivariate polyunomials and constant functions, so that we can take partial derivatives of polynomials like
```C++
auto expr = x*x*y+x*y*y*c(4.)+y*c(1.)+x;
```

## In C++14
We start by describing the C++14 original example. The features which make the example "C++14" are mainly the use of "auto" and "constexpr", both introduced from C++11 and made more usable since C++14. in a nutshell The former is (very handy) syntactic sugar and could be avoided with some extra coding, the latter is a language feature which allows to reuse the same code for both compile-time and run-time computations.
We define a placeholder for the independent variable "x", which is implementing the identity function, and instantiate a global variable "x" in order to compy with the our target syntax

```C++
struct p {

    constexpr p(){};

    template <typename T>
    constexpr T operator()( T t_ ) const
    {
        return t_;
    }
};

constexpr auto x = p();
```

We also define the expressions "plus" and "times": function objects whose evaluation in turn evaluates 
the template arguments and returns their sum or multiplication respectively. 
We also overload the operators "+" and "*" to meet the target API.

```C++
template <typename T1, typename T2>
struct expr_plus {

    template <typename T>
    constexpr auto operator()( T t_ ) const
    {
        return T1()( t_ ) + T2()( t_ );
    }
};

template <typename T1, typename T2>
constexpr expr_plus<T1, T2>
operator+( T1 arg1, T2 arg2 )
{
    return expr_plus<T1, T2>();
}

template <typename T1, typename T2>
struct expr_times {

    template <typename T>
    constexpr auto operator()( T t_ ) const
    {
	return T1()( t_ ) * T2()( t_ );
    }
};

template <typename T1, typename T2>
constexpr expr_times<T1, T2>
operator*( T1 arg1, T2 arg2 )
{
    return expr_times<T1, T2>();
}
```
Notice that there's no data members. The object functions representing the operations are stateless, 
and all the information needed to parse an expression is contained in its type (which might be limiting, but for this simple example it's ok).
So far so good, we can evaluate expressions at compile time or run time

```C++
int main(){
    constexpr auto expr = x*x+x;
    static_assert(expr(6.)==42.);
}
```
We implemented a simple expression template idiom in modern C++, written in few lines 
of code and with a pretty cool API. Let's add the symbolic derivation to it. 
We know how the derivative of a sum and of a multiplication look like, 
we know that the derivative of x is 1, and the derivative of a constant is 0.
We create an expression for the derivative with a specialization for all the aforementioned cases, plus
the case in which we take the derivative of a derivative expression.

```C++
template <typename T1>
struct expr_derivative {
    using value_t = int;

    template <typename T>
    constexpr T operator()( T t_ ) const
    {
     	return 0;
    }
};

template <>
struct expr_derivative<p> {

    using value_t = int;

    template <typename T>
    constexpr auto operator()( T t_ ) const
    {
	return (T)1;
    }
};

template <typename T1, typename T2>
struct expr_derivative<expr_plus<T1, T2>> {

    using value_t = decltype( D( T1() ) + D( T2() ) );

    template <typename T>
    constexpr auto operator()( T t_ ) const
    {
        return value_t()( t_ );
    }
};

template <typename T1, typename T2>
struct expr_derivative<expr_times<T1, T2>> {

    using value_t = decltype( T1() * D( T2() )
                              + D( T1() ) * T2() );

    template <typename T>
    constexpr auto operator()( T t_ ) const
    {
     	return value_t()( t_ );
    }
};
```

We left out the double derivative, which is more tricky since it must recursively call itself. You might have observed the definition of the 
"value_t" type in the expressions above, which seems useless. It turns out to be necessary now
```C++
template <typename T1>
struct expr_derivative<expr_derivative<T1>> {

    using value_t = expr_derivative<typename expr_derivative<T1>::value_t>;

    template <typename T>
    constexpr auto operator()( T t_ ) const
    {
        return value_t()( t_ );
    }
};
```
without using the type value_t type, and just returning
```
expr_derivative<expr_derivative<T1>>(t)
```
we would end up in an infinite recursion, while appending ::value_t we write the expression of the symbolic first derivative
in terms of its fundamental components, so that it can be fed to the next expr_derivative.

Eventually we define our function "D" returning an instance of the derivative expression:

```C++
template <typename T1>
constexpr expr_derivative<T1>
D( T1 arg1 )
{
    return expr_derivative<T1>();
}
```

We have now the full code for automatic differentiation of polynomials.

```C++
int main(){
    constexpr auto expr = D(D(x*x+x));
    static_assert(expr(6.)==42.);
}
```
## In C++17

With C++17, using in particular costexpr generic lambdas and constexpr if, functional programming becomes easier 
and more readabe in C++. We cannot replace completely the template metaprogramming needed by 
the expression template idiom used in the example above, because when we compute the derivatives 
of an expression we still need to "parse" it, and there is not (yet) such thing like a constexpr 
parser. However we can replace many of the templated objects
defined above with constexpr generic lambdas. Below there is an equivalent code snippet, which 
uses the C++17 generic lambdas feature.

We define the function objects "plus" and "times", with convenient overloads of the corresponding operators 
"+" and "*", in a simila way as done for
the previous example. These cannot be defined as constexpr lambdas for one main reasons: we need to access the type of the 
two arguments passed to the lambdas. More in detail, we can think of a generic lambda conceptually "as if" it was a functor like:

```C++
struct UniqueName{
       /*constructor initializing captured arguments*/
       template<typename T1, typename T2>
       operator(T1 t1, T2  t2){ /*body*/ }
       /*captured member arguments*/
}
```

so there's no way to extract the types T1 and T2 from the lambda type. Ok, not a big deal, we define our structs

```C++
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
```

Now we define our independent variable x as a generic lambda representing the identity function, and we define constants
as functions "c" returning a given value regardless the input parameter.

```C++
    constexpr auto x = [](auto t){ return t ;};
    constexpr auto c = [](auto t){ return  [t](auto){return t;};};
```

We can express univariate polynomial functions like

```C++
    auto poly = c(1.) +x*c(3.)+ x*x*c(2.);
```

Easy. Let's see how to define differentation.
We need to set a unity and a zero elements as we did in the previous example:

```C++
    constexpr auto zero = [](auto t){ return 0 ;};
    constexpr auto one = [](auto t){ return 1 ;};
```

then we can define the rules of differentiation by using a "recursive lambda" function.
This is a bit tricky, because one cannot call a generic lambda function inside the body of the generic lambda itself,
because the lambda is not defined yet (it's type is "incomplete"). So we have to use a trick to do that: 
we define a lambda calling another lambda

```C++
constexpr auto recursive = [](auto t, auto self){
	  if(!/*stop condition*/)
	  self(t, self); //recursion
}
```

Then we express the recursion when the lambda function is already defined,
by passing the recursive function as argutment of its own call.

```C++
constexpr auto D = [](){ recursive(t, recursive) }
```

The implementation of the derivation is 

```C++
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
```

## Extension

We have shown so far how to build a simple Automatic Differentation idiom with C++14 and C++17 in about 60 lines of code. 
What ifwe want to generalize the example further, including multivariate functions 
(i.e. with more than one independent variable)?
For instance

```C++
auto expr = x*x*y+y*y*x;
```

and take its derivatives with respect to x and y

```C++
auto dx = Dx(expr);
auto dy = Dy(expr);
static assert(dx(5.)(5.)==dy(5.)(5.)), "error);
```

We can do this with very little effort by substituting what we consider now a "value" with a univariate function,
obtaining thus functions of functions (or high order functions, in functional programming terminology).

We interpret a multivariate function as a function of functions (a high order function) so that we can
 bind one of the variables to a value leaving the other free:
the function

```C++
auto f = dy(5.);
```

becomes a univariate function, and we can perform opearations on this partially evaluated function too:

```C++
auto g = f+x(5.);
```

In order to achieve that we change first of all the definition of our variable x, transforming it into a second order function which returns the first argument. 
We define another independent variable "y" which returns the second argument instead, while constants "c"
return a value which is independent of both the arguments: 

```C++
    constexpr auto x = [](auto t){ return [t](auto t2){return t ;};};
    constexpr auto y = [](auto t){ return [t](auto t2){return t2;};};
    constexpr auto c = [](auto t){ return [t](auto){ return [t](auto){return t;};};};
```

Also the "one" and "zero" elements in our algebra must be second order functions, so we change their definition to

```C++
    constexpr auto zero = [](auto t){ return [t](auto t2){return 0 ;};};
    constexpr auto one = [](auto t){ return [t](auto t2){return 1 ;};};
```

We have to do some changes to the recursive lambda which computes the derivatives too. We have to 
pass on a tag, specifying the variable with respect to which we are taking the derivative (x or y),
and we have to specify a case in our "constexpr switch" for the derivative of "y".

```C++
    using same = std::is_same<typename std::decay<T>::type,typename std::decay<U>::type>;
    constexpr auto recursion = [one, zero, x, y](auto tag, auto t, auto self){
        using t_t = decltype(t);
        using x_t = decltype(x);
        using y_t = decltype(y);
        using tag_t = decltype(tag);
        if constexpr (same<t_t,x_t>::value)
        if constexpr (same<tag_t,x_t>::value)
                return one;
            else
                return zero;
        else if constexpr(same<t_t, y_t>::value)
            if constexpr (same<tag_t,y_t>::value)
                return one;
            else
                return zero;
        else if constexpr( is_plus<typename std::decay<decltype(t)>::type>::value)
            return plus{self(tag, t.t1, self), self(tag, t.t2, self)};
        else if constexpr( is_times<typename std::decay<decltype(t)>::type>::value)
            return plus{times{self(tag, t.t1, self), t.t2}, times{t.t1, self(tag, t.t2, self)}};
        else
            return zero;
    };

    constexpr auto Dx = [recursion,x](auto t){return recursion(x, t, recursion);};
    constexpr auto Dy = [recursion,y](auto t){return recursion(y, t, recursion);};
```

Ad that's it, all the operator defined for the previous example don't need any change, and we can
 compute derivatives of multivariate functions as

```C++
    auto constexpr ex = Dy(x*x*c(4.)+y*x*c(4.));
    auto constexpr val=ex(5.)(1.);
```
