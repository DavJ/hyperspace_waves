"""
Biquaternion class implementation.

A biquaternion is a quaternion with complex coefficients, used in the
Unified Biquaternion Theory (UBT) framework.

Author: David Jaros
Site: www.octonion-multiverse.com
"""

import numpy as np
import cmath


class Biquaternion:
    """
    Represents a biquaternion q = w + xi + yj + zk where w, x, y, z are complex numbers.
    
    Quaternion units satisfy: i² = j² = k² = ijk = -1
    """
    
    def __init__(self, w=0+0j, x=0+0j, y=0+0j, z=0+0j):
        """
        Initialize a biquaternion.
        
        Args:
            w: scalar (complex) component
            x: i component (complex)
            y: j component (complex)
            z: k component (complex)
        """
        self.w = complex(w)
        self.x = complex(x)
        self.y = complex(y)
        self.z = complex(z)
    
    def __repr__(self):
        return f"Biquaternion({self.w}, {self.x}, {self.y}, {self.z})"
    
    def __str__(self):
        return f"{self.w} + {self.x}i + {self.y}j + {self.z}k"
    
    def __add__(self, other):
        """Add two biquaternions or add a scalar."""
        if isinstance(other, Biquaternion):
            return Biquaternion(
                self.w + other.w,
                self.x + other.x,
                self.y + other.y,
                self.z + other.z
            )
        else:
            # Scalar addition
            return Biquaternion(self.w + other, self.x, self.y, self.z)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        """Subtract two biquaternions or subtract a scalar."""
        if isinstance(other, Biquaternion):
            return Biquaternion(
                self.w - other.w,
                self.x - other.x,
                self.y - other.y,
                self.z - other.z
            )
        else:
            return Biquaternion(self.w - other, self.x, self.y, self.z)
    
    def __rsub__(self, other):
        """Right subtraction (scalar - biquaternion)."""
        return Biquaternion(other - self.w, -self.x, -self.y, -self.z)
    
    def __mul__(self, other):
        """
        Multiply two biquaternions using quaternion multiplication rules.
        
        Hamilton's rules:
        i² = j² = k² = ijk = -1
        ij = k, jk = i, ki = j
        ji = -k, kj = -i, ik = -j
        """
        if isinstance(other, Biquaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Biquaternion(w, x, y, z)
        else:
            # Scalar multiplication
            return Biquaternion(self.w * other, self.x * other, self.y * other, self.z * other)
    
    def __rmul__(self, other):
        """Right multiplication (scalar * biquaternion)."""
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Divide by a scalar or biquaternion."""
        if isinstance(other, Biquaternion):
            # q1 / q2 = q1 * q2^(-1)
            return self * other.inverse()
        else:
            # Scalar division
            return Biquaternion(self.w / other, self.x / other, self.y / other, self.z / other)
    
    def conjugate(self):
        """Return the quaternion conjugate: q* = w - xi - yj - zk"""
        return Biquaternion(self.w, -self.x, -self.y, -self.z)
    
    def complex_conjugate(self):
        """Return the complex conjugate of all components."""
        return Biquaternion(
            self.w.conjugate(),
            self.x.conjugate(),
            self.y.conjugate(),
            self.z.conjugate()
        )
    
    def full_conjugate(self):
        """Return both quaternion and complex conjugate."""
        return Biquaternion(
            self.w.conjugate(),
            -self.x.conjugate(),
            -self.y.conjugate(),
            -self.z.conjugate()
        )
    
    def norm_squared(self):
        """
        Return |q|² = |w|² + |x|² + |y|² + |z|² 
        where |·| denotes the complex absolute value (modulus).
        """
        return (abs(self.w)**2 + abs(self.x)**2 + 
                abs(self.y)**2 + abs(self.z)**2)
    
    def norm(self):
        """Return the norm |q|."""
        return np.sqrt(self.norm_squared())
    
    def inverse(self):
        """Return the multiplicative inverse q^(-1) = q*/|q|²."""
        n2 = self.norm_squared()
        if n2 == 0:
            raise ValueError("Cannot invert zero biquaternion")
        return self.conjugate() / n2
    
    def exp(self):
        """
        Compute the exponential of a biquaternion.
        
        For q = w + v⃗ (where v⃗ = xi + yj + zk):
        exp(q) = exp(w) * (cos(|v⃗|) + v⃗/|v⃗| * sin(|v⃗|))
        """
        v_norm = np.sqrt(abs(self.x)**2 + abs(self.y)**2 + abs(self.z)**2)
        exp_w = cmath.exp(self.w)
        
        if v_norm < 1e-10:
            # Small vector part, use Taylor expansion
            return Biquaternion(exp_w, exp_w * self.x, exp_w * self.y, exp_w * self.z)
        
        cos_v = cmath.cos(v_norm)
        sin_v = cmath.sin(v_norm)
        factor = sin_v / v_norm
        
        return Biquaternion(
            exp_w * cos_v,
            exp_w * factor * self.x,
            exp_w * factor * self.y,
            exp_w * factor * self.z
        )
    
    def to_array(self):
        """Convert to numpy array [w, x, y, z]."""
        return np.array([self.w, self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr):
        """Create biquaternion from array [w, x, y, z]."""
        return cls(arr[0], arr[1], arr[2], arr[3])
    
    def scalar_part(self):
        """Return the scalar (real quaternion) part."""
        return self.w
    
    def vector_part(self):
        """Return the vector part as a tuple (x, y, z)."""
        return (self.x, self.y, self.z)
    
    def trace(self):
        """Return the trace (scalar part) of the biquaternion."""
        return self.w


def bq_exp(q):
    """Compute exponential of a biquaternion (convenience function)."""
    if isinstance(q, Biquaternion):
        return q.exp()
    else:
        # Assume scalar
        return cmath.exp(q)
