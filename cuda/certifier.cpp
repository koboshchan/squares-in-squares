// Fast multithreaded C++17 certifier with exact decimal + interval escalation parity
// against src/squares/_official_metric.py (including post-roundtrip verification).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cert {

constexpr int MAX_N = 1000;
constexpr size_t MAX_LITERAL_CHARS = 100;
constexpr double MAX_COORD = 1e4;
constexpr double MAX_ANGLE = 100.0;
constexpr double AABB_PAD = 1e-7;
constexpr double UNDECIDED_TOL = 1e-8;

// Fixed-point signed number with 1 integer limb (up to +-10^9) and FL fractional limbs in base 10^9.
// Each fractional limb holds 9 decimal digits, so FL=10 -> 90 decimal digits, FL=60 -> 540 decimal digits.
template <int FL>
struct Fix {
    static constexpr uint64_t BASE = 1000000000ULL;
    int sign = 1; // +1 or -1 (zero has sign +1)
    uint64_t w[FL + 1] = {0}; // w[0] is integer part, w[1..FL] are fractional limbs (most significant first)

    bool is_zero() const {
        for (int i = 0; i <= FL; ++i) if (w[i] != 0) return false;
        return true;
    }
    void normalize_zero() {
        if (is_zero()) sign = 1;
    }
    static Fix from_int(long long v) {
        Fix r;
        if (v < 0) { r.sign = -1; v = -v; }
        r.w[0] = (uint64_t)v;
        return r;
    }
    double to_double() const {
        double v = (double)w[0];
        double scale = 1.0;
        for (int i = 1; i <= std::min(FL, 4); ++i) {
            scale *= 1e-9;
            v += (double)w[i] * scale;
        }
        return sign < 0 ? -v : v;
    }
    std::string format_50_decimals() const {
        // Round half-to-even at the 50th decimal digit.
        // 50th decimal digit is the 5th digit of w[6] (since w[1..5] = 45 digits, w[6] = digits 46..54).
        // Specifically, w[6] / 10000 gives digits 46..50, and w[6] % 10000 gives digits 51..54.
        uint64_t rem_top = w[6] % 10000ULL;
        bool tail_nonzero = false;
        for (int i = 7; i <= FL; ++i) if (w[i] != 0) { tail_nonzero = true; break; }
        uint64_t q6 = w[6] / 10000ULL; // 0..99999
        bool round_up = false;
        if (rem_top > 5000ULL || (rem_top == 5000ULL && tail_nonzero)) {
            round_up = true;
        } else if (rem_top == 5000ULL && !tail_nonzero) {
            round_up = (q6 & 1ULL) != 0;
        }
        uint64_t d[7] = {w[0], w[1], w[2], w[3], w[4], w[5], q6};
        if (round_up) {
            d[6] += 1;
            if (d[6] >= 100000ULL) {
                d[6] -= 100000ULL;
                for (int i = 5; i >= 1; --i) {
                    d[i] += 1;
                    if (d[i] >= BASE) d[i] -= BASE;
                    else goto carry_done;
                }
                d[0] += 1;
            }
        }
    carry_done:
        char buf[128];
        bool all_z = (d[0] == 0 && d[1] == 0 && d[2] == 0 && d[3] == 0 && d[4] == 0 && d[5] == 0 && d[6] == 0);
        const char* pfx = (sign < 0 && !all_z) ? "-" : "";
        std::snprintf(
            buf, sizeof(buf),
            "%s%llu.%09llu%09llu%09llu%09llu%09llu%05llu",
            pfx,
            (unsigned long long)d[0],
            (unsigned long long)d[1],
            (unsigned long long)d[2],
            (unsigned long long)d[3],
            (unsigned long long)d[4],
            (unsigned long long)d[5],
            (unsigned long long)d[6]
        );
        return std::string(buf);
    }
};

template <int FL>
int cmp_abs(const Fix<FL>& a, const Fix<FL>& b) {
    for (int i = 0; i <= FL; ++i) {
        if (a.w[i] != b.w[i]) return (a.w[i] < b.w[i]) ? -1 : 1;
    }
    return 0;
}

template <int FL>
int cmp_val(const Fix<FL>& a, const Fix<FL>& b) {
    if (a.is_zero() && b.is_zero()) return 0;
    if (a.sign != b.sign) return (a.sign < b.sign) ? -1 : 1;
    int c = cmp_abs(a, b);
    return (a.sign > 0) ? c : -c;
}

template <int FL>
Fix<FL> add_abs(const Fix<FL>& a, const Fix<FL>& b, int sgn) {
    Fix<FL> r;
    r.sign = sgn;
    uint64_t carry = 0;
    for (int i = FL; i >= 1; --i) {
        uint64_t s = a.w[i] + b.w[i] + carry;
        if (s >= Fix<FL>::BASE) { s -= Fix<FL>::BASE; carry = 1; } else carry = 0;
        r.w[i] = s;
    }
    r.w[0] = a.w[0] + b.w[0] + carry;
    r.normalize_zero();
    return r;
}

template <int FL>
Fix<FL> sub_abs(const Fix<FL>& a, const Fix<FL>& b, int sgn) {
    // Assumes |a| >= |b|
    Fix<FL> r;
    r.sign = sgn;
    int64_t borrow = 0;
    for (int i = FL; i >= 1; --i) {
        int64_t d = (int64_t)a.w[i] - (int64_t)b.w[i] - borrow;
        if (d < 0) { d += (int64_t)Fix<FL>::BASE; borrow = 1; } else borrow = 0;
        r.w[i] = (uint64_t)d;
    }
    r.w[0] = (uint64_t)((int64_t)a.w[0] - (int64_t)b.w[0] - borrow);
    r.normalize_zero();
    return r;
}

template <int FL>
Fix<FL> add(const Fix<FL>& a, const Fix<FL>& b) {
    if (a.sign == b.sign) return add_abs(a, b, a.sign);
    int c = cmp_abs(a, b);
    if (c == 0) return Fix<FL>();
    return (c > 0) ? sub_abs(a, b, a.sign) : sub_abs(b, a, b.sign);
}

template <int FL>
Fix<FL> sub(const Fix<FL>& a, const Fix<FL>& b) {
    Fix<FL> nb = b;
    nb.sign = -b.sign;
    nb.normalize_zero();
    return add(a, nb);
}

template <int FL>
void add_1ulp_abs(Fix<FL>& a) {
    uint64_t carry = 1;
    for (int i = FL; i >= 1 && carry; --i) {
        uint64_t s = a.w[i] + carry;
        if (s >= Fix<FL>::BASE) { a.w[i] = s - Fix<FL>::BASE; carry = 1; }
        else { a.w[i] = s; carry = 0; }
    }
    if (carry) a.w[0] += carry;
}

// Multiply a * b with optional directed rounding: dir = 0 (nearest/trunc), -1 (down), +1 (up)
template <int FL>
Fix<FL> mul(const Fix<FL>& a, const Fix<FL>& b, int dir = 0) {
    if (a.is_zero() || b.is_zero()) return Fix<FL>();
    uint64_t prod[2 * FL + 2] = {0};
    for (int i = 0; i <= FL; ++i) {
        if (a.w[i] == 0) continue;
        uint64_t carry = 0;
        for (int j = FL; j >= 0; --j) {
            __uint128_t cur = (__uint128_t)prod[i + j] + (__uint128_t)a.w[i] * b.w[j] + carry;
            prod[i + j] = (uint64_t)(cur % Fix<FL>::BASE);
            carry = (uint64_t)(cur / Fix<FL>::BASE);
        }
        int k = i - 1;
        while (carry > 0 && k >= 0) {
            uint64_t s = prod[k] + carry;
            prod[k] = s % Fix<FL>::BASE;
            carry = s / Fix<FL>::BASE;
            --k;
        }
    }
    Fix<FL> r;
    r.sign = a.sign * b.sign;
    for (int i = 0; i <= FL; ++i) r.w[i] = prod[i];
    bool rem_nonzero = false;
    for (int i = FL + 1; i <= 2 * FL; ++i) if (prod[i] != 0) { rem_nonzero = true; break; }
    if (dir == 0) {
        if (prod[FL + 1] >= Fix<FL>::BASE / 2) add_1ulp_abs(r);
    } else if (rem_nonzero) {
        if ((dir > 0 && r.sign > 0) || (dir < 0 && r.sign < 0)) {
            add_1ulp_abs(r);
        }
    }
    r.normalize_zero();
    return r;
}

template <int FL>
Fix<FL> div_small_int(const Fix<FL>& a, uint64_t d, int dir = 0) {
    Fix<FL> r;
    r.sign = a.sign;
    uint64_t rem = 0;
    for (int i = 0; i <= FL; ++i) {
        __uint128_t cur = (__uint128_t)rem * Fix<FL>::BASE + a.w[i];
        r.w[i] = (uint64_t)(cur / d);
        rem = (uint64_t)(cur % d);
    }
    if (dir == 0) {
        if (rem * 2 >= d) add_1ulp_abs(r);
    } else if (rem != 0) {
        if ((dir > 0 && r.sign > 0) || (dir < 0 && r.sign < 0)) add_1ulp_abs(r);
    }
    r.normalize_zero();
    return r;
}

template <int FL>
struct Iv {
    Fix<FL> lo, hi;
    static Iv point(const Fix<FL>& v) { return {v, v}; }
    Iv operator+(const Iv& o) const { return {add(lo, o.lo), add(hi, o.hi)}; }
    Iv operator-(const Iv& o) const { return {sub(lo, o.hi), sub(hi, o.lo)}; }
    Iv operator-() const {
        Fix<FL> nlo = hi, nhi = lo;
        nlo.sign = -hi.sign; nlo.normalize_zero();
        nhi.sign = -lo.sign; nhi.normalize_zero();
        return {nlo, nhi};
    }
    Iv operator*(const Iv& o) const {
        Fix<FL> d1 = mul(lo, o.lo, -1), u1 = mul(lo, o.lo, +1);
        Fix<FL> d2 = mul(lo, o.hi, -1), u2 = mul(lo, o.hi, +1);
        Fix<FL> d3 = mul(hi, o.lo, -1), u3 = mul(hi, o.lo, +1);
        Fix<FL> d4 = mul(hi, o.hi, -1), u4 = mul(hi, o.hi, +1);
        Fix<FL> mn = d1, mx = u1;
        if (cmp_val(d2, mn) < 0) mn = d2;
        if (cmp_val(d3, mn) < 0) mn = d3;
        if (cmp_val(d4, mn) < 0) mn = d4;
        if (cmp_val(u2, mx) > 0) mx = u2;
        if (cmp_val(u3, mx) > 0) mx = u3;
        if (cmp_val(u4, mx) > 0) mx = u4;
        return {mn, mx};
    }
};

// Parse decimal string into Fix<FL> (with directed rounding: dir=0 nearest, -1 down, +1 up)
template <int FL>
Fix<FL> parse_decimal_fix(const std::string& raw, int dir = 0) {
    size_t i = 0, n = raw.size();
    while (i < n && std::isspace((unsigned char)raw[i])) ++i;
    while (n > i && std::isspace((unsigned char)raw[n - 1])) --n;
    int sgn = 1;
    if (i < n && (raw[i] == '+' || raw[i] == '-')) {
        if (raw[i] == '-') sgn = -1;
        ++i;
    }
    std::string digits;
    digits.reserve(n - i);
    long long frac_digits = 0;
    bool seen_dot = false;
    while (i < n && raw[i] != 'e' && raw[i] != 'E') {
        if (raw[i] == '.') {
            seen_dot = true;
        } else {
            digits.push_back(raw[i]);
            if (seen_dot) ++frac_digits;
        }
        ++i;
    }
    long long exp10 = 0;
    if (i < n && (raw[i] == 'e' || raw[i] == 'E')) {
        ++i;
        int esgn = 1;
        if (i < n && (raw[i] == '+' || raw[i] == '-')) {
            if (raw[i] == '-') esgn = -1;
            ++i;
        }
        while (i < n && std::isdigit((unsigned char)raw[i])) {
            exp10 = exp10 * 10 + (raw[i] - '0');
            ++i;
        }
        exp10 *= esgn;
    }
    long long int_len = (long long)digits.size() - frac_digits + exp10;
    Fix<FL> r;
    r.sign = sgn;
    // Extract integer part
    uint64_t int_val = 0;
    for (long long k = 0; k < int_len; ++k) {
        char c = (k >= 0 && k < (long long)digits.size()) ? digits[(size_t)k] : '0';
        int_val = int_val * 10 + (uint64_t)(c - '0');
    }
    r.w[0] = int_val;
    // Extract fractional limbs (each limb is 9 decimal digits)
    for (int limb = 1; limb <= FL; ++limb) {
        uint64_t v = 0;
        for (int d = 0; d < 9; ++d) {
            long long pos = int_len + (long long)(limb - 1) * 9 + d;
            char c = (pos >= 0 && pos < (long long)digits.size()) ? digits[(size_t)pos] : '0';
            v = v * 10 + (uint64_t)(c - '0');
        }
        r.w[limb] = v;
    }
    bool rem_nonzero = false;
    long long rem_start = std::max(0LL, int_len + (long long)FL * 9);
    for (long long pos = rem_start; pos < (long long)digits.size(); ++pos) {
        if (digits[(size_t)pos] != '0') { rem_nonzero = true; break; }
    }
    if (rem_nonzero && dir != 0) {
        if ((dir > 0 && sgn > 0) || (dir < 0 && sgn < 0)) add_1ulp_abs(r);
    }
    r.normalize_zero();
    return r;
}

static bool is_dyadic_decimal_str(const std::string& raw) {
    size_t i = 0, n = raw.size();
    while (i < n && std::isspace((unsigned char)raw[i])) ++i;
    while (n > i && std::isspace((unsigned char)raw[n - 1])) --n;
    if (i < n && (raw[i] == '+' || raw[i] == '-')) ++i;
    std::string digits;
    long long frac_digits = 0;
    bool seen_dot = false;
    while (i < n && raw[i] != 'e' && raw[i] != 'E') {
        if (raw[i] == '.') seen_dot = true;
        else { digits.push_back(raw[i]); if (seen_dot) ++frac_digits; }
        ++i;
    }
    long long exp10 = 0;
    if (i < n && (raw[i] == 'e' || raw[i] == 'E')) {
        ++i;
        int esgn = 1;
        if (i < n && (raw[i] == '+' || raw[i] == '-')) { if (raw[i] == '-') esgn = -1; ++i; }
        while (i < n && std::isdigit((unsigned char)raw[i])) { exp10 = exp10 * 10 + (raw[i] - '0'); ++i; }
        exp10 *= esgn;
    }
    long long eff_frac = frac_digits - exp10;
    while (eff_frac > 0 && !digits.empty() && digits.back() == '0') {
        digits.pop_back();
        --eff_frac;
    }
    if (eff_frac <= 0) return true;
    if (eff_frac > 25) return false;
    // Check if digits is divisible by 5^eff_frac
    for (long long k = 0; k < eff_frac; ++k) {
        if (digits.empty()) return true;
        int last = digits.back() - '0';
        if (last % 5 != 0) return false;
        // Divide digits by 5
        int rem = 0;
        std::string next_d;
        next_d.reserve(digits.size());
        for (char c : digits) {
            int cur = rem * 10 + (c - '0');
            int q = cur / 5;
            rem = cur % 5;
            if (!next_d.empty() || q != 0) next_d.push_back((char)('0' + q));
        }
        digits = std::move(next_d);
    }
    return true;
}

template <int FL>
Iv<FL> parse_decimal_iv(const std::string& raw) {
    Fix<FL> lo = parse_decimal_fix<FL>(raw, -1);
    Fix<FL> hi = parse_decimal_fix<FL>(raw, +1);
    if (!is_dyadic_decimal_str(raw)) {
        Fix<FL> ulp;
        ulp.w[FL] = 1;
        lo = sub(lo, ulp);
        hi = add(hi, ulp);
    }
    return {lo, hi};
}

// 540-digit constant for pi/2
static const char* PI_HALF_STR =
    "1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585339"
    "9107404325664115332354692230477529111586267970406424055872514205135096926055277982231147447746519098"
    "2214405487832966723064237824116893391582635600954572824283161565386415559563044484210977355059400485"
    "7492389293099245510655716316154220989218499846888949675083019800051100514872368941154883214599416258"
    "7630228031340772652010253145121415210968393898212262598912242342639728019727481121132833606270987663"
    "3184560871636462872546121833290233099787539786414835847812261143213250625359595866490484368352035968";

static const char* TWO_PI_STR =
    "6.2831853071795864769252867665590057683943387987502116419498891846156328125724179972560696506842341359"
    "6429617302656461329418768921910116446345071881625696223490056820540387704221111928924589790986076392"
    "8857621951331866892256951296467573566330542403818291297132646261545662238252177936843909420237601942"
    "9969557172396982042622865264616883956873999387555798700332079200204402059489475764619532858397665035"
    "0520912125363090608041012580485660843873575592848905039564896937055891207890992448453133442508395065"
    "2738243486545851490184487333160932399150159145659343391249044572853002501438383465961937473408143872";

// Compute sin and cos of x (in [-pi/4, pi/4]) via Taylor series
template <int FL>
void sincos_reduced_fix(const Fix<FL>& x, Fix<FL>& s_out, Fix<FL>& c_out, int max_terms) {
    Fix<FL> x2 = mul(x, x, 0);
    Fix<FL> c_sum = Fix<FL>::from_int(1);
    Fix<FL> c_term = Fix<FL>::from_int(1);
    Fix<FL> s_sum = x;
    Fix<FL> s_term = x;
    for (int m = 1; m <= max_terms; ++m) {
        c_term = mul(c_term, x2, 0);
        c_term = div_small_int(c_term, (uint64_t)(2 * m - 1) * (uint64_t)(2 * m), 0);
        c_term.sign = (m & 1) ? -1 : 1;
        c_term.normalize_zero();
        c_sum = add(c_sum, c_term);

        s_term = mul(s_term, x2, 0);
        s_term = div_small_int(s_term, (uint64_t)(2 * m) * (uint64_t)(2 * m + 1), 0);
        s_term.sign = (m & 1) ? -x.sign : x.sign;
        s_term.normalize_zero();
        s_sum = add(s_sum, s_term);
        if (c_term.is_zero() && s_term.is_zero()) break;
    }
    s_out = s_sum;
    c_out = c_sum;
}

template <int FL>
void sincos_fix(const Fix<FL>& angle, Fix<FL>& s_out, Fix<FL>& c_out) {
    static const Fix<FL> pi_half = parse_decimal_fix<FL>(PI_HALF_STR, 0);
    double approx_q = std::round(angle.to_double() / 1.57079632679489661923);
    long long q = (long long)approx_q;
    Fix<FL> q_fix = Fix<FL>::from_int(q);
    Fix<FL> r = sub(angle, mul(q_fix, pi_half, 0));
    int max_terms = (FL <= 10) ? 32 : 115;
    Fix<FL> sr, cr;
    sincos_reduced_fix(r, sr, cr, max_terms);
    int oct = (int)((q % 4 + 4) % 4);
    if (oct == 0) { s_out = sr; c_out = cr; }
    else if (oct == 1) { s_out = cr; c_out = sr; c_out.sign = -c_out.sign; c_out.normalize_zero(); }
    else if (oct == 2) {
        s_out = sr; s_out.sign = -s_out.sign; s_out.normalize_zero();
        c_out = cr; c_out.sign = -c_out.sign; c_out.normalize_zero();
    } else {
        s_out = cr; s_out.sign = -s_out.sign; s_out.normalize_zero();
        c_out = sr;
    }
}

template <int FL>
void sincos_iv(const Iv<FL>& angle, Iv<FL>& s_out, Iv<FL>& c_out) {
    if (angle.lo.is_zero() && angle.hi.is_zero()) {
        s_out = Iv<FL>::point(Fix<FL>::from_int(0));
        c_out = Iv<FL>::point(Fix<FL>::from_int(1));
        return;
    }
    static const Iv<FL> pi_half = parse_decimal_iv<FL>(PI_HALF_STR);
    double mid = 0.5 * (angle.lo.to_double() + angle.hi.to_double());
    long long q = (long long)std::round(mid / 1.57079632679489661923);
    Iv<FL> q_iv = Iv<FL>::point(Fix<FL>::from_int(q));
    Iv<FL> r = angle - (q_iv * pi_half);
    Fix<FL> s_lo, c_lo, s_hi, c_hi;
    int max_terms = (FL <= 8) ? 28 : (FL <= 15) ? 45 : 145;
    sincos_reduced_fix(r.lo, s_lo, c_lo, max_terms);
    if (cmp_val(r.lo, r.hi) == 0) {
        s_hi = s_lo;
        c_hi = c_lo;
    } else {
        sincos_reduced_fix(r.hi, s_hi, c_hi, max_terms);
    }
    auto widen = [](Fix<FL> a, Fix<FL> b, bool can_cross_zero_peak) -> Iv<FL> {
        Fix<FL> mn = (cmp_val(a, b) < 0) ? a : b;
        Fix<FL> mx = (cmp_val(a, b) > 0) ? a : b;
        Fix<FL> eps;
        eps.w[FL] = 1;
        mn = sub(mn, eps);
        mx = add(mx, eps);
        if (can_cross_zero_peak) {
            Fix<FL> one = Fix<FL>::from_int(1);
            if (cmp_val(mx, one) < 0) mx = one;
        }
        return {mn, mx};
    };
    bool crosses_zero = (r.lo.sign < 0 && r.hi.sign > 0) || r.lo.is_zero() || r.hi.is_zero();
    Iv<FL> sr = widen(s_lo, s_hi, false);
    Iv<FL> cr = widen(c_lo, c_hi, crosses_zero);
    int oct = (int)((q % 4 + 4) % 4);
    if (oct == 0) { s_out = sr; c_out = cr; }
    else if (oct == 1) { s_out = cr; c_out = -sr; }
    else if (oct == 2) { s_out = -sr; c_out = -cr; }
    else { s_out = -cr; c_out = sr; }
}

static bool is_valid_decimal(const std::string& raw, std::string& stripped) {
    size_t a = 0, b = raw.size();
    while (a < b && std::isspace((unsigned char)raw[a])) ++a;
    while (b > a && std::isspace((unsigned char)raw[b - 1])) --b;
    stripped = raw.substr(a, b - a);
    if (stripped.empty() || stripped.size() > MAX_LITERAL_CHARS) return false;
    size_t i = 0, n = stripped.size();
    if (stripped[i] == '+' || stripped[i] == '-') ++i;
    if (i >= n) return false;
    bool has_digits = false;
    while (i < n && std::isdigit((unsigned char)stripped[i])) { has_digits = true; ++i; }
    if (i < n && stripped[i] == '.') {
        ++i;
        while (i < n && std::isdigit((unsigned char)stripped[i])) { has_digits = true; ++i; }
    }
    if (!has_digits) return false;
    if (i < n && (stripped[i] == 'e' || stripped[i] == 'E')) {
        ++i;
        if (i < n && (stripped[i] == '+' || stripped[i] == '-')) ++i;
        bool exp_digits = false;
        while (i < n && std::isdigit((unsigned char)stripped[i])) { exp_digits = true; ++i; }
        if (!exp_digits) return false;
    }
    return i == n;
}

struct RowData {
    int square_id;
    std::string x, y, angle;
};

struct Square {
    double x, y, angle;
    std::string x_str, y_str, angle_str;
    double corners[4][2];
    double aabb[4];
};

static void compute_corners_and_aabb(Square& sq) {
    double c = std::cos(sq.angle), s = std::sin(sq.angle);
    double dx1 = 0.5 * c - 0.5 * s, dy1 = 0.5 * s + 0.5 * c;
    double dx2 = -0.5 * c - 0.5 * s, dy2 = -0.5 * s + 0.5 * c;
    sq.corners[0][0] = sq.x + dx1; sq.corners[0][1] = sq.y + dy1;
    sq.corners[1][0] = sq.x + dx2; sq.corners[1][1] = sq.y + dy2;
    sq.corners[2][0] = sq.x - dx1; sq.corners[2][1] = sq.y - dy1;
    sq.corners[3][0] = sq.x - dx2; sq.corners[3][1] = sq.y - dy2;
    double min_x = sq.corners[0][0], max_x = sq.corners[0][0];
    double min_y = sq.corners[0][1], max_y = sq.corners[0][1];
    for (int k = 1; k < 4; ++k) {
        min_x = std::min(min_x, sq.corners[k][0]);
        max_x = std::max(max_x, sq.corners[k][0]);
        min_y = std::min(min_y, sq.corners[k][1]);
        max_y = std::max(max_y, sq.corners[k][1]);
    }
    sq.aabb[0] = min_x - AABB_PAD;
    sq.aabb[1] = min_y - AABB_PAD;
    sq.aabb[2] = max_x + AABB_PAD;
    sq.aabb[3] = max_y + AABB_PAD;
}

static int sat_check_float(const Square& sq1, const Square& sq2) {
    // Returns 1 = separated, -1 = overlapping, 0 = undecided
    double c1 = std::cos(sq1.angle), s1 = std::sin(sq1.angle);
    double c2 = std::cos(sq2.angle), s2 = std::sin(sq2.angle);
    double axes[4][2] = {{c1, s1}, {-s1, c1}, {c2, s2}, {-s2, c2}};
    double min_overlap = 1e300;
    for (int a = 0; a < 4; ++a) {
        double ax = axes[a][0], ay = axes[a][1];
        double min1 = 1e300, max1 = -1e300, min2 = 1e300, max2 = -1e300;
        for (int k = 0; k < 4; ++k) {
            double p1 = ax * sq1.corners[k][0] + ay * sq1.corners[k][1];
            double p2 = ax * sq2.corners[k][0] + ay * sq2.corners[k][1];
            if (p1 < min1) min1 = p1;
            if (p1 > max1) max1 = p1;
            if (p2 < min2) min2 = p2;
            if (p2 > max2) max2 = p2;
        }
        double overlap = std::min(max1, max2) - std::max(min1, min2);
        if (overlap < -UNDECIDED_TOL) return 1;
        if (overlap < min_overlap) min_overlap = overlap;
    }
    if (min_overlap > UNDECIDED_TOL) return -1;
    return 0;
}

template <int FL>
struct SqIvCache {
    bool ready = false;
    bool axis_aligned = false;
    Iv<FL> c, s;
    Iv<FL> min_x, max_x, min_y, max_y;
    Iv<FL> cor[4][2];
};

template <int FL>
void ensure_sq_iv(
    int idx, const std::vector<Square>& squares, const Iv<FL>& x0_iv, const Iv<FL>& y0_iv,
    std::vector<SqIvCache<FL>>& cache,
    std::unordered_map<std::string, std::pair<Iv<FL>, Iv<FL>>>& trig_map
) {
    if (cache[idx].ready) return;
    const Square& sq = squares[idx];
    Iv<FL> cx = parse_decimal_iv<FL>(sq.x_str) - x0_iv;
    Iv<FL> cy = parse_decimal_iv<FL>(sq.y_str) - y0_iv;
    auto it = trig_map.find(sq.angle_str);
    if (it != trig_map.end()) {
        cache[idx].c = it->second.first;
        cache[idx].s = it->second.second;
    } else {
        Iv<FL> a = parse_decimal_iv<FL>(sq.angle_str);
        sincos_iv(a, cache[idx].s, cache[idx].c);
        trig_map.emplace(sq.angle_str, std::make_pair(cache[idx].c, cache[idx].s));
    }
    static const Iv<FL> half = parse_decimal_iv<FL>("0.5");
    const Iv<FL>& c = cache[idx].c;
    const Iv<FL>& s = cache[idx].s;
    cache[idx].axis_aligned = (s.lo.is_zero() && s.hi.is_zero());
    if (cache[idx].axis_aligned) {
        cache[idx].min_x = cx - half;
        cache[idx].max_x = cx + half;
        cache[idx].min_y = cy - half;
        cache[idx].max_y = cy + half;
    }
    Iv<FL> dx1 = half * c - half * s, dy1 = half * s + half * c;
    Iv<FL> dx2 = -half * c - half * s, dy2 = -half * s + half * c;
    cache[idx].cor[0][0] = cx + dx1; cache[idx].cor[0][1] = cy + dy1;
    cache[idx].cor[1][0] = cx + dx2; cache[idx].cor[1][1] = cy + dy2;
    cache[idx].cor[2][0] = cx - dx1; cache[idx].cor[2][1] = cy - dy1;
    cache[idx].cor[3][0] = cx - dx2; cache[idx].cor[3][1] = cy - dy2;
    cache[idx].ready = true;
}

template <int FL>
int sat_check_interval_cached(const SqIvCache<FL>& sq1, const SqIvCache<FL>& sq2) {
    // Returns 1 = certified separated, -1 = certified overlapping, 0 = undecided
    Fix<FL> zero;
    if (sq1.axis_aligned && sq2.axis_aligned) {
        Iv<FL> ox = {
            (cmp_val(sq1.max_x.lo, sq2.max_x.lo) < 0 ? sq1.max_x.lo : sq2.max_x.lo),
            (cmp_val(sq1.max_x.hi, sq2.max_x.hi) < 0 ? sq1.max_x.hi : sq2.max_x.hi)
        };
        Iv<FL> mx = {
            (cmp_val(sq1.min_x.lo, sq2.min_x.lo) > 0 ? sq1.min_x.lo : sq2.min_x.lo),
            (cmp_val(sq1.min_x.hi, sq2.min_x.hi) > 0 ? sq1.min_x.hi : sq2.min_x.hi)
        };
        Iv<FL> ovx = ox - mx;
        if (cmp_val(ovx.hi, zero) <= 0) return 1;

        Iv<FL> oy = {
            (cmp_val(sq1.max_y.lo, sq2.max_y.lo) < 0 ? sq1.max_y.lo : sq2.max_y.lo),
            (cmp_val(sq1.max_y.hi, sq2.max_y.hi) < 0 ? sq1.max_y.hi : sq2.max_y.hi)
        };
        Iv<FL> my = {
            (cmp_val(sq1.min_y.lo, sq2.min_y.lo) > 0 ? sq1.min_y.lo : sq2.min_y.lo),
            (cmp_val(sq1.min_y.hi, sq2.min_y.hi) > 0 ? sq1.min_y.hi : sq2.min_y.hi)
        };
        Iv<FL> ovy = oy - my;
        if (cmp_val(ovy.hi, zero) <= 0) return 1;
        if (cmp_val(ovx.lo, zero) <= 0 || cmp_val(ovy.lo, zero) <= 0) return 0;
        return -1;
    }
    Iv<FL> axes[4][2] = {{sq1.c, sq1.s}, {-sq1.s, sq1.c}, {sq2.c, sq2.s}, {-sq2.s, sq2.c}};
    bool has_undecided = false;
    for (int a = 0; a < 4; ++a) {
        Iv<FL> ax = axes[a][0], ay = axes[a][1];
        Iv<FL> min1 = ax * sq1.cor[0][0] + ay * sq1.cor[0][1], max1 = min1;
        Iv<FL> min2 = ax * sq2.cor[0][0] + ay * sq2.cor[0][1], max2 = min2;
        for (int k = 1; k < 4; ++k) {
            Iv<FL> p1 = ax * sq1.cor[k][0] + ay * sq1.cor[k][1];
            Iv<FL> p2 = ax * sq2.cor[k][0] + ay * sq2.cor[k][1];
            if (cmp_val(p1.lo, min1.lo) < 0) min1.lo = p1.lo;
            if (cmp_val(p1.hi, min1.hi) < 0) min1.hi = p1.hi;
            if (cmp_val(p1.lo, max1.lo) > 0) max1.lo = p1.lo;
            if (cmp_val(p1.hi, max1.hi) > 0) max1.hi = p1.hi;

            if (cmp_val(p2.lo, min2.lo) < 0) min2.lo = p2.lo;
            if (cmp_val(p2.hi, min2.hi) < 0) min2.hi = p2.hi;
            if (cmp_val(p2.lo, max2.lo) > 0) max2.lo = p2.lo;
            if (cmp_val(p2.hi, max2.hi) > 0) max2.hi = p2.hi;
        }
        Iv<FL> min_max = {
            (cmp_val(max1.lo, max2.lo) < 0) ? max1.lo : max2.lo,
            (cmp_val(max1.hi, max2.hi) < 0) ? max1.hi : max2.hi
        };
        Iv<FL> max_min = {
            (cmp_val(min1.lo, min2.lo) > 0) ? min1.lo : min2.lo,
            (cmp_val(min1.hi, min2.hi) > 0) ? min1.hi : min2.hi
        };
        Iv<FL> overlap = min_max - max_min;
        if (cmp_val(overlap.hi, zero) <= 0) {
            return 1; // Certified separated on this axis
        }
        if (cmp_val(overlap.lo, zero) <= 0) {
            has_undecided = true;
        }
    }
    if (has_undecided) return 0;
    return -1; // Certified overlapping on all 4 axes
}

template <int FL>
int run_escalation_stage(
    int n, const std::vector<Square>& squares,
    const std::vector<std::pair<int, int>>& in_pairs,
    std::vector<std::pair<int, int>>& out_undecided
) {
    Iv<FL> x0 = parse_decimal_iv<FL>(squares[0].x_str);
    Iv<FL> y0 = parse_decimal_iv<FL>(squares[0].y_str);
    std::vector<SqIvCache<FL>> cache(n);
    std::unordered_map<std::string, std::pair<Iv<FL>, Iv<FL>>> trig_map;
    out_undecided.clear();
    for (auto [u, v] : in_pairs) {
        ensure_sq_iv<FL>(u, squares, x0, y0, cache, trig_map);
        ensure_sq_iv<FL>(v, squares, x0, y0, cache, trig_map);
        int r = sat_check_interval_cached<FL>(cache[u], cache[v]);
        if (r < 0) return -1;
        if (r == 0) out_undecided.emplace_back(u, v);
    }
    return out_undecided.empty() ? 1 : 0;
}

struct PuzzleResult {
    int n = 0;
    double score = 0.0;
    std::string L;
    std::string reason = "missing from submission";
    bool ok = false;
};

static double infeasible_score(int n) {
    return std::ceil(std::sqrt((double)n)) / std::sqrt((double)n) - 1.0;
}

static PuzzleResult evaluate_single_puzzle(int n, const std::vector<RowData>& rows) {
    PuzzleResult res;
    res.n = n;
    res.score = infeasible_score(n);
    if ((int)rows.size() != n) {
        res.reason = "n=" + std::to_string(n) + ": expected " + std::to_string(n) + " rows, got " + std::to_string(rows.size());
        return res;
    }
    std::vector<const RowData*> by_id(n, nullptr);
    for (const auto& r : rows) {
        if (r.square_id >= 0 && r.square_id < n && by_id[r.square_id] == nullptr) {
            by_id[r.square_id] = &r;
        } else {
            res.reason = "n=" + std::to_string(n) + ": square_ids should be 0.." + std::to_string(n - 1);
            return res;
        }
    }

    // Precision FL=9 (81 fractional decimal digits > 60-digit L_DPS)
    using F60 = Fix<9>;
    static const F60 two_pi_60 = parse_decimal_fix<9>(TWO_PI_STR, 0);
    static const F60 half_60 = parse_decimal_fix<9>("0.5", 0);

    std::vector<Square> squares(n);
    F60 x0_mp, y0_mp;

    for (int sid = 0; sid < n; ++sid) {
        std::string xs, ys, as;
        if (!is_valid_decimal(by_id[sid]->x, xs) ||
            !is_valid_decimal(by_id[sid]->y, ys) ||
            !is_valid_decimal(by_id[sid]->angle, as)) {
            res.reason = "n=" + std::to_string(n) + ", square " + std::to_string(sid) + ": cannot parse coordinates";
            return res;
        }
        char* endp = nullptr;
        double xf = std::strtod(xs.c_str(), &endp);
        double yf = std::strtod(ys.c_str(), &endp);
        double af = std::strtod(as.c_str(), &endp);
        if (!std::isfinite(xf) || !std::isfinite(yf) || !std::isfinite(af)) {
            res.reason = "n=" + std::to_string(n) + ", square " + std::to_string(sid) + ": value out of range for a double";
            return res;
        }
        if (std::fabs(xf) > MAX_COORD || std::fabs(yf) > MAX_COORD) {
            res.reason = "n=" + std::to_string(n) + ", square " + std::to_string(sid) + ": coordinate out of bounds";
            return res;
        }
        if (std::fabs(af) > MAX_ANGLE) {
            res.reason = "n=" + std::to_string(n) + ", square " + std::to_string(sid) + ": angle out of bounds";
            return res;
        }
        squares[sid].x_str = xs;
        squares[sid].y_str = ys;
        squares[sid].angle_str = as;
    }

    x0_mp = parse_decimal_fix<9>(squares[0].x_str, 0);
    y0_mp = parse_decimal_fix<9>(squares[0].y_str, 0);

    // Cache high-precision (cos, sin) by angle_str within the puzzle
    std::unordered_map<std::string, std::pair<F60, F60>> trig_cache;
    trig_cache.reserve(32);

    F60 min_x_mp, max_x_mp, min_y_mp, max_y_mp;
    bool first_corner = true;

    for (int sid = 0; sid < n; ++sid) {
        F60 dx_mp = sub(parse_decimal_fix<9>(squares[sid].x_str, 0), x0_mp);
        F60 dy_mp = sub(parse_decimal_fix<9>(squares[sid].y_str, 0), y0_mp);
        F60 ang_mp = parse_decimal_fix<9>(squares[sid].angle_str, 0);

        // Floor modulo 2*pi
        long long k2pi = (long long)std::floor(ang_mp.to_double() / 6.283185307179586476925);
        F60 rem_ang = sub(ang_mp, mul(F60::from_int(k2pi), two_pi_60, 0));
        if (rem_ang.sign < 0 && !rem_ang.is_zero()) rem_ang = add(rem_ang, two_pi_60);
        if (cmp_val(rem_ang, two_pi_60) >= 0) rem_ang = sub(rem_ang, two_pi_60);

        squares[sid].x = dx_mp.to_double();
        squares[sid].y = dy_mp.to_double();
        squares[sid].angle = rem_ang.to_double();
        compute_corners_and_aabb(squares[sid]);

        F60 s_mp, c_mp;
        if (squares[sid].angle_str == "0" || squares[sid].angle_str == "0.0" || ang_mp.is_zero()) {
            c_mp = F60::from_int(1);
            s_mp = F60::from_int(0);
        } else {
            auto it = trig_cache.find(squares[sid].angle_str);
            if (it != trig_cache.end()) {
                c_mp = it->second.first;
                s_mp = it->second.second;
            } else {
                sincos_fix<9>(ang_mp, s_mp, c_mp);
                trig_cache.emplace(squares[sid].angle_str, std::make_pair(c_mp, s_mp));
            }
        }
        F60 hc = mul(half_60, c_mp, 0), hs = mul(half_60, s_mp, 0);
        F60 dx1 = sub(hc, hs), dy1 = add(hs, hc);
        F60 dx2 = sub( F60::from_int(0), add(hc, hs) ), dy2 = sub(hc, hs);
        F60 cx[4] = {add(dx_mp, dx1), add(dx_mp, dx2), sub(dx_mp, dx1), sub(dx_mp, dx2)};
        F60 cy[4] = {add(dy_mp, dy1), add(dy_mp, dy2), sub(dy_mp, dy1), sub(dy_mp, dy2)};
        for (int k = 0; k < 4; ++k) {
            if (first_corner) {
                min_x_mp = max_x_mp = cx[k];
                min_y_mp = max_y_mp = cy[k];
                first_corner = false;
            } else {
                if (cmp_val(cx[k], min_x_mp) < 0) min_x_mp = cx[k];
                if (cmp_val(cx[k], max_x_mp) > 0) max_x_mp = cx[k];
                if (cmp_val(cy[k], min_y_mp) < 0) min_y_mp = cy[k];
                if (cmp_val(cy[k], max_y_mp) > 0) max_y_mp = cy[k];
            }
        }
    }

    F60 span_x = sub(max_x_mp, min_x_mp);
    F60 span_y = sub(max_y_mp, min_y_mp);
    F60 s_mp = (cmp_val(span_x, span_y) >= 0) ? span_x : span_y;

    // Sort and sweep AABBs
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return squares[a].aabb[0] < squares[b].aabb[0];
    });

    std::vector<std::pair<int, int>> undecided_pairs;
    for (int ii = 0; ii < n; ++ii) {
        int i = order[ii];
        const double* ai = squares[i].aabb;
        for (int jj = ii + 1; jj < n; ++jj) {
            int j = order[jj];
            const double* aj = squares[j].aabb;
            if (aj[0] > ai[2]) break;
            if (!(ai[2] < aj[0] || aj[2] < ai[0] || ai[3] < aj[1] || aj[3] < ai[1])) {
                int u = std::min(i, j), v = std::max(i, j);
                int sc = sat_check_float(squares[u], squares[v]);
                if (sc < 0) {
                    res.reason = "infeasible: Squares " + std::to_string(u) + " and " + std::to_string(v) + " overlap (float check)";
                    return res;
                }
                if (sc == 0) {
                    undecided_pairs.emplace_back(u, v);
                }
            }
        }
    }

    if (!undecided_pairs.empty()) {
        std::vector<std::pair<int, int>> rem1, rem2;
        int r1 = run_escalation_stage<7>(n, squares, undecided_pairs, rem1);
        if (r1 < 0) { res.reason = "infeasible: Squares overlap (certified)"; return res; }
        if (r1 == 0) {
            int r2 = run_escalation_stage<15>(n, squares, rem1, rem2);
            if (r2 < 0) { res.reason = "infeasible: Squares overlap (certified)"; return res; }
            if (r2 == 0) {
                std::vector<std::pair<int, int>> rem3;
                int r3 = run_escalation_stage<58>(n, squares, rem2, rem3);
                if (r3 < 0) { res.reason = "infeasible: Squares overlap (certified)"; return res; }
                if (r3 == 0) { res.reason = "infeasible: Overlap undecidable at max precision"; return res; }
            }
        }
    }

    double s_f = s_mp.to_double();
    double L_B = std::sqrt((double)n);
    if (s_f < L_B - 1e-7) {
        res.reason = "infeasible: L is below area lower bound";
        return res;
    }
    res.ok = true;
    res.reason = "ok";
    res.L = s_mp.format_50_decimals();
    res.score = std::max(0.0, s_f / L_B - 1.0);
    return res;
}

static std::string roundtrip_float_str(const std::string& s) {
    char* endp = nullptr;
    double v = std::strtod(s.c_str(), &endp);
    char buf[64];
    for (int prec = 15; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    bool has_dot_or_exp = false;
    for (int i = 0; buf[i]; ++i) {
        if (buf[i] == '.' || buf[i] == 'e' || buf[i] == 'E') { has_dot_or_exp = true; break; }
    }
    std::string out(buf);
    if (!has_dot_or_exp) out += ".0";
    return out;
}

} // namespace cert

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: certifier <submission.csv> [--max-n N] [--threads T] [--roundtrip] [--details out.json]\n";
        return 2;
    }
    std::string csv_path = argv[1];
    int max_n = 1000;
    int num_threads = (int)std::max(1u, std::thread::hardware_concurrency());
    bool do_roundtrip = false;
    std::string details_path;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--max-n" && i + 1 < argc) max_n = std::atoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) num_threads = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--roundtrip") do_roundtrip = true;
        else if (arg == "--details" && i + 1 < argc) details_path = argv[++i];
    }

    auto t0 = std::chrono::steady_clock::now();
    std::ifstream in(csv_path);
    if (!in) {
        std::cerr << "Cannot open " << csv_path << "\n";
        return 2;
    }
    std::string line;
    if (!std::getline(in, line)) {
        std::cerr << "Empty CSV\n";
        return 2;
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "id,n,square_id,x,y,angle") {
        std::cerr << "CSV header must be id,n,square_id,x,y,angle\n";
        return 2;
    }

    std::vector<std::vector<cert::RowData>> puzzles(max_n + 1);
    long long expected_id = 0;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        // Fast comma split
        size_t p1 = line.find(',');
        if (p1 == std::string::npos) return 2;
        size_t p2 = line.find(',', p1 + 1);
        if (p2 == std::string::npos) return 2;
        size_t p3 = line.find(',', p2 + 1);
        if (p3 == std::string::npos) return 2;
        size_t p4 = line.find(',', p3 + 1);
        if (p4 == std::string::npos) return 2;
        size_t p5 = line.find(',', p4 + 1);
        if (p5 == std::string::npos || line.find(',', p5 + 1) != std::string::npos) return 2;

        long long id = std::strtoll(line.c_str(), nullptr, 10);
        if (id != expected_id) {
            std::cerr << "Row id mismatch at " << expected_id << "\n";
            return 2;
        }
        ++expected_id;
        int n = std::atoi(line.c_str() + p1 + 1);
        int sid = std::atoi(line.c_str() + p2 + 1);
        if (n < 1 || n > cert::MAX_N) return 2;
        if (n <= max_n) {
            cert::RowData rd{
                sid,
                line.substr(p3 + 1, p4 - p3 - 1),
                line.substr(p4 + 1, p5 - p4 - 1),
                line.substr(p5 + 1)
            };
            if (do_roundtrip) {
                rd.x = cert::roundtrip_float_str(rd.x);
                rd.y = cert::roundtrip_float_str(rd.y);
                rd.angle = cert::roundtrip_float_str(rd.angle);
            }
            puzzles[n].push_back(std::move(rd));
        }
    }

    std::vector<cert::PuzzleResult> results(max_n + 1);
    std::atomic<int> next_n{max_n};

    auto worker = [&]() {
        while (true) {
            int n = next_n.fetch_sub(1, std::memory_order_relaxed);
            if (n < 1) break;
            if (puzzles[n].empty()) {
                results[n].n = n;
                results[n].score = cert::infeasible_score(n);
                results[n].reason = "missing from submission";
                results[n].ok = false;
            } else {
                results[n] = cert::evaluate_single_puzzle(n, puzzles[n]);
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    double total_score = 0.0;
    int accepted = 0, missing = 0, rejected = 0;
    for (int n = 1; n <= max_n; ++n) {
        total_score += results[n].score;
        if (results[n].ok) ++accepted;
        else if (results[n].reason == "missing from submission") ++missing;
        else ++rejected;
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    if (!details_path.empty()) {
        std::ofstream out(details_path);
        out << std::setprecision(17)
            << "{\"score\":" << total_score
            << ",\"max_n\":" << max_n
            << ",\"accepted\":" << accepted
            << ",\"missing\":" << missing
            << ",\"rejected\":" << rejected
            << ",\"seconds\":" << sec
            << ",\"threads\":" << num_threads
            << ",\"roundtrip\":" << (do_roundtrip ? "true" : "false")
            << ",\"results\":[";
        for (int n = 1; n <= max_n; ++n) {
            if (n > 1) out << ",";
            out << "{\"n\":" << n
                << ",\"score\":" << results[n].score
                << ",\"L\":" << (results[n].ok ? ("\"" + results[n].L + "\"") : "null")
                << ",\"reason\":\"" << results[n].reason << "\"}";
        }
        out << "]}\n";
    }

    std::cout << std::setprecision(17)
              << "{\"score\":" << total_score
              << ",\"max_n\":" << max_n
              << ",\"accepted\":" << accepted
              << ",\"missing\":" << missing
              << ",\"rejected\":" << rejected
              << ",\"seconds\":" << sec
              << ",\"threads\":" << num_threads
              << ",\"roundtrip\":" << (do_roundtrip ? "true" : "false")
              << "}\n";
    return 0;
}
