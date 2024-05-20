#Based on https://github.com/dougthor42/PyErf
#Same thing except jitted

# ---------------------------------------------------------------------------
### Imports
# ---------------------------------------------------------------------------
# Standard Library
import math

import numba as nb
from numba import njit,prange

# ---------------------------------------------------------------------------
### Constants
# ---------------------------------------------------------------------------
# While some of these are used only in _ndtri, we don't want to
# calculate them each time a user calls erfinv. So we define them at the
# module level and they'll only be calculated once.
PI = math.pi
ROOT_2PI = math.sqrt(2 * PI)
EXP_NEG2 = math.exp(-2)

# math.inf was added in Python 3.5.
try:
    from math import inf
except ImportError:
    inf = float('inf')


#: Inputs above this value are considered infinity.
MAXVAL = 1e50

@njit(fastmath=False)
def _ndtri(y):
    """
    Port of cephes ``ndtri.c``: inverse normal distribution function.

    See https://github.com/jeremybarnes/cephes/blob/master/cprob/ndtri.c
    """
    # approximation for 0 <= abs(z - 0.5) <= 3/8
    P0 = [
        -5.99633501014107895267E1,
        9.80010754185999661536E1,
        -5.66762857469070293439E1,
        1.39312609387279679503E1,
        -1.23916583867381258016E0,
    ]

    Q0 = [
        1.95448858338141759834E0,
        4.67627912898881538453E0,
        8.63602421390890590575E1,
        -2.25462687854119370527E2,
        2.00260212380060660359E2,
        -8.20372256168333339912E1,
        1.59056225126211695515E1,
        -1.18331621121330003142E0,
    ]

    # Approximation for interval z = sqrt(-2 log y ) between 2 and 8
    # i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
    P1 = [
        4.05544892305962419923E0,
        3.15251094599893866154E1,
        5.71628192246421288162E1,
        4.40805073893200834700E1,
        1.46849561928858024014E1,
        2.18663306850790267539E0,
        -1.40256079171354495875E-1,
        -3.50424626827848203418E-2,
        -8.57456785154685413611E-4,
    ]

    Q1 = [
        1.57799883256466749731E1,
        4.53907635128879210584E1,
        4.13172038254672030440E1,
        1.50425385692907503408E1,
        2.50464946208309415979E0,
        -1.42182922854787788574E-1,
        -3.80806407691578277194E-2,
        -9.33259480895457427372E-4,
    ]

    # Approximation for interval z = sqrt(-2 log y ) between 8 and 64
    # i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
    P2 = [
        3.23774891776946035970E0,
        6.91522889068984211695E0,
        3.93881025292474443415E0,
        1.33303460815807542389E0,
        2.01485389549179081538E-1,
        1.23716634817820021358E-2,
        3.01581553508235416007E-4,
        2.65806974686737550832E-6,
        6.23974539184983293730E-9,
    ]

    Q2 = [
        6.02427039364742014255E0,
        3.67983563856160859403E0,
        1.37702099489081330271E0,
        2.16236993594496635890E-1,
        1.34204006088543189037E-2,
        3.28014464682127739104E-4,
        2.89247864745380683936E-6,
        6.79019408009981274425E-9,
    ]

    sign_flag = 1

    if y > (1 - EXP_NEG2):
        y = 1 - y
        sign_flag = 0

    # Shortcut case where we don't need high precision
    # between -0.135 and 0.135
    if y > EXP_NEG2:
        y -= 0.5
        y2 = y ** 2
        x = y + y * (y2 * _polevl(y2, P0, 4) / _p1evl(y2, Q0, 8))
        x = x * ROOT_2PI
        return x

    x = math.sqrt(-2.0 * math.log(y))
    x0 = x - math.log(x) / x

    z = 1.0 / x
    if x < 8.0:                 # y > exp(-32) = 1.2664165549e-14
        x1 = z * _polevl(z, P1, 8) / _p1evl(z, Q1, 8)
    else:
        x1 = z * _polevl(z, P2, 8) / _p1evl(z, Q2, 8)

    x = x0 - x1
    if sign_flag != 0:
        x = -x

    return x


@njit(fastmath=False)
def _polevl(x, coefs, N):
    """
    Port of cephes ``polevl.c``: evaluate polynomial

    See https://github.com/jeremybarnes/cephes/blob/master/cprob/polevl.c
    """
    ans = 0
    power = len(coefs) - 1
    for coef in coefs:
        try:
            ans += coef * x**power
        except:
            pass
        power -= 1
    return ans


@njit(fastmath=False)
def _p1evl(x, coefs, N):
    """
    Port of cephes ``polevl.c``: evaluate polynomial, assuming coef[N] = 1

    See https://github.com/jeremybarnes/cephes/blob/master/cprob/polevl.c
    """
    return _polevl(x, [1] + coefs, N)


@njit(fastmath=False)
def erfinv(z):
    """
    Calculate the inverse error function at point ``z``.

    This is a direct port of the SciPy ``erfinv`` function, originally
    written in C.

    Parameters
    ----------
    z : numeric

    Returns
    -------
    float

    References
    ----------
    + https://en.wikipedia.org/wiki/Error_function#Inverse_functions
    + http://functions.wolfram.com/GammaBetaErf/InverseErf/

    Examples
    --------
    >>> round(erfinv(0.1), 12)
    0.088855990494
    >>> round(erfinv(0.5), 12)
    0.476936276204
    >>> round(erfinv(-0.5), 12)
    -0.476936276204
    >>> round(erfinv(0.95), 12)
    1.38590382435
    >>> round(erf(erfinv(0.3)), 3)
    0.3
    >>> round(erfinv(erf(0.5)), 3)
    0.5
    >>> erfinv(0)
    0
    >>> erfinv(1)
    inf
    >>> erfinv(-1)
    -inf
    """
    if abs(z) > 1:
        raise ValueError("`z` must be between -1 and 1 inclusive")

    # Shortcut special cases
    if z == 0:
        return 0
    if z == 1:
        return inf
    if z == -1:
        return -inf

    # otherwise calculate things.
    return _ndtri((z + 1) / 2.0) / math.sqrt(2)
