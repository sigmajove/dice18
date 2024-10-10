// Program to design 18-sided dice

// Make sure M_PI gets defined
#define _USE_MATH_DEFINES

#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

static constexpr double EPSILON = 1.0e-10;

// Returns a random double value x where 0 < x < 1
double Rand01() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<> dis(0.0, 1.0);
  for (;;) {
    double result = dis(gen);
    if (0.0 < result && result < 1.0) {
      return result;
    }
  }
}

// A vector in 3-space.
class Vector {
 public:
  Vector(double dx, double dy, double dz) : dx_(dx), dy_(dy), dz_(dz) {}

  double dx() const { return dx_; }
  double dy() const { return dy_; }
  double dz() const { return dz_; }

  std::string Format() const {
    return std::format("({}, {}, {})", dx_, dy_, dz_);
  }

 private:
  double dx_;
  double dy_;
  double dz_;
};

// See https://en.wikipedia.org/wiki/Dot_product
double DotProduct(const Vector& a, const Vector& b) {
  return a.dx() * b.dx() + a.dy() * b.dy() + a.dz() * b.dz();
}

// See https://en.wikipedia.org/wiki/Cross_product
Vector CrossProduct(const Vector& v0, const Vector& v1) {
  return Vector(v0.dy() * v1.dz() - v0.dz() * v1.dy(),
                v0.dz() * v1.dx() - v0.dx() * v1.dz(),
                v0.dx() * v1.dy() - v0.dy() * v1.dx());
}

// The half-space (a, b, c, d) is defined by the equation
// a*x + b*y + c*z - d <= 0.
using HalfSpace = std::tuple<double, double, double, double>;

// Return a vector that is perpendicular to the boundary plane of
// h, pointing out of the half-space.
Vector Normal(const HalfSpace& h) {
  return Vector(std::get<0>(h), std::get<1>(h), std::get<2>(h));
}

// Returns true of a and b are within a floating point roundoff error
// from each other.  Borrowed from Python
bool IsClose(double a, double b) {
  return std::abs(a - b) <=
         std::max(1.0e-09 * std::max(std::abs(a), std::abs(b)), 1.5e-14);
}

// A Vertex is defined by its Cartesian coordinates in 3-space.
class Vertex {
 public:
  Vertex(double x, double y, double z) : x_(x), y_(y), z_(z) {}

  bool IsClose(const Vertex& other) const {
    return ::IsClose(x_, other.x_) && ::IsClose(y_, other.y_) &&
           ::IsClose(z_, other.z_);
  }

  double x() const { return x_; }
  double y() const { return y_; }
  double z() const { return z_; }

  std::string Format() const { return std::format("({}, {}, {})", x_, y_, z_); }

 private:
  double x_;
  double y_;
  double z_;
};

// Find the line that is the intersection of the two planes.
// The line is represented by point on the line and a vector
// parallel to the line.
std::optional<std::pair<Vertex, Vector>> PlaneIntersection(
    const HalfSpace& plane1, const HalfSpace& plane2) {
  // The vector we will return.
  const Vector vector = CrossProduct(Normal(plane1), Normal(plane2));

  // Find a point on the intersection of the two planes.
  // We set one of x, y, or z to zero. Then we have two linear
  // equations in two variables, which we can solve straightforwardly.
  // We pick the variable to set to zero that maximizes the
  // magnitude of the divisor in the computation of the point.
  const double adx = std::abs(vector.dx());
  const double ady = std::abs(vector.dy());
  const double adz = std::abs(vector.dz());
  const double m = std::max({adx, ady, adz});
  if (m < EPSILON) {
    // The two planes are (nearly) parallel; there is no intersection to return.
    return std::nullopt;
  }

  const auto [a1, b1, c1, d1] = plane1;
  const auto [a2, b2, c2, d2] = plane2;

  double x, y, z;
  if (m == adx) {
    x = 0.0;
    y = (c1 * d2 - c2 * d1) / vector.dx();
    z = (b2 * d1 - b1 * d2) / vector.dx();
  } else if (m == ady) {
    x = (c2 * d1 - c1 * d2) / vector.dy();
    y = 0.0;
    z = (a1 * d2 - a2 * d1) / vector.dy();
  } else {  // m == adz
    x = (b1 * d2 - b2 * d1) / vector.dz();
    y = (a2 * d1 - a1 * d2) / vector.dz();
    z = 0.0;
  }

  return std::make_pair(Vertex(x, y, z), vector);
}

// Given a plane and a line, find the point (if any) where they intersect.
std::optional<Vertex> PlaneLine(const HalfSpace& plane,
                                const std::pair<Vertex, Vector>& line) {
  const Vector& vector = line.second;
  const double denom = DotProduct(Normal(plane), vector);
  if (std::abs(denom) < EPSILON) {
    // The plane and line are parallel; there is no point to return.
    return std::nullopt;
  }
  const Vertex& vertex = line.first;
  const auto [a, b, c, d] = plane;
  const double t =
      (a * vertex.x() + b * vertex.y() + c * vertex.z() + d) / denom;
  return Vertex(vertex.x() - t * vector.dx(), vertex.y() - t * vector.dy(),
                vertex.z() - t * vector.dz());
}

// Return the point (if any) where three planes intersect.
std::optional<Vertex>
ThreePlanes(const HalfSpace& p0, const HalfSpace &p1, const HalfSpace &p2) {
  const auto& xyz = PlaneIntersection(p0, p1);
  if (xyz) {
    return PlaneLine(p2, *xyz);
  }
  return std::nullopt;
}

std::vector<HalfSpace> GenHalves() {
  std::vector<HalfSpace> result;
  result.reserve(18);
  for (int i = 0; i < 9; ++i) {
    // Generate a random point on the unit sphere.
    // See https://mathworld.wolfram.com/SpherePointPicking.html
    const double theta = 2 * M_PI * Rand01();
    const double phi = std::acos(2 * Rand01() - 1);
    const double sin_phi = std::sin(phi);
    const double x = std::cos(theta) * sin_phi;
    const double y = std::sin(theta) * sin_phi;
    const double z = std::cos(phi);

    // Push the half spaces defined by the point and its antipode.
    result.push_back(HalfSpace(x, y, z, -1.0));
    result.push_back(HalfSpace(-x, -y, -z, -1.0));
  }
  return result;
}

class VertexMap {
 public:
  void Insert(const Vertex& v, const std::set<std::size_t>& faces);

  const std::vector<std::pair<Vertex, std::set<std::size_t>>>& map() {
    return map_;
  }

 private:
  std::vector<std::pair<Vertex, std::set<std::size_t>>> map_;
};

void VertexMap::Insert(const Vertex& v, const std::set<std::size_t>& faces) {
  for (auto& [key, value] : map_) {
    if (key.IsClose(v)) {
      value.insert(faces.begin(), faces.end());
      return;
    }
  }
  map_.emplace_back(v, faces);
}

int main() {
  const HalfSpace p0 = std::make_tuple(0, 0, 1, -5);  // z == 5
  const HalfSpace p1 = std::make_tuple(0, 1, 0, -7);  // y == 7
  const HalfSpace p2 = std::make_tuple(1, 0, 0, -8);  // x == 8
  const auto result = ThreePlanes(p2, p0, p1);
  if (result) {
    std::cout << result.value().Format() << "\n";
  } else {
    std::cout << "parallel";
  }
}
