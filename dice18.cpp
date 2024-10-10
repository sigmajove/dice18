// Program to design 18-sided dice

// Make sure M_PI gets defined
#define _USE_MATH_DEFINES

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <format>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
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

class HalfSpace {
 public:
  // The half-space (a, b, c, d) is defined by the equation
  // a*x + b*y + c*z - d <= 0.
  HalfSpace(double a, double b, double c, double d)
      : a_(a), b_(b), c_(c), d_(d) {}

  std::tuple<double, double, double, double> Parameters() const {
    return std::make_tuple(a_, b_, c_, d_);
  } 

  // Returns a vector that is perpendicular to the boundary plane,
  // pointing out of the half-space.
  Vector Normal() const { return Vector(a_, b_, c_); }

  // Normal defines the orientation of the boundary plane, and Position
  // defines its position.
  double Position() const { return d_; }

  std::string Format() const {
    return std::format("({}, {}, {}, {})", a_, b_, c_, d_);
  }

 private:
  double a_;
  double b_;
  double c_;
  double d_;
};

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

  friend Vector operator-(const Vertex& lhs, const Vertex& rhs) {
    return Vector(lhs.x_ - rhs.x_, lhs.y_ - rhs.y_, lhs.z_ - rhs.z_);
  }

  // Converts the Vertex to a Vector.
  // It's the vector from the origin (0, 0, 0) the Vertex.
  Vector ToVector() const { return Vector(x_, y_, z_); }

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
  const Vector vector = CrossProduct(plane1.Normal(), plane2.Normal());

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

  const auto [a1, b1, c1, d1] = plane1.Parameters();
  const auto [a2, b2, c2, d2] = plane2.Parameters();

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
  const double denom = DotProduct(plane.Normal(), vector);
  if (std::abs(denom) < EPSILON) {
    // The plane and line are parallel; there is no point to return.
    return std::nullopt;
  }
  const Vertex& vertex = line.first;
  const auto [a, b, c, d] = plane.Parameters();
  const double t =
      (a * vertex.x() + b * vertex.y() + c * vertex.z() + d) / denom;
  return Vertex(vertex.x() - t * vector.dx(), vertex.y() - t * vector.dy(),
                vertex.z() - t * vector.dz());
}

// Return the point (if any) where three planes intersect.
std::optional<Vertex> ThreePlanes(const HalfSpace& p0, const HalfSpace& p1,
                                  const HalfSpace& p2) {
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
  using KeyVal = std::pair<Vertex, std::set<std::size_t>>;

  std::size_t size() const { return map_.size(); }
  const KeyVal& get(std::size_t i) const { return map_[i]; }

  void Insert(const Vertex& v, const std::set<std::size_t>& faces);

  // Removes every map entry filter(key_val) returns true.
  void Filter(std::function<bool(const KeyVal&)> filter) {
    map_.erase(std::remove_if(map_.begin(), map_.end(), filter), map_.end());
  }

  const std::vector<KeyVal>& map() const { return map_; }

 private:
  std::vector<KeyVal> map_;
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

// Given three vertices v0, v1, v2 of a polygon in the border plane for half,
// return whether the are in counterclockwise order, as viewed from outside
// the half-space.
bool IsCounterclockwise(const Vertex& v0, const Vertex& v1, const Vertex& v2,
                        const HalfSpace half) {
  return DotProduct(CrossProduct(v1 - v0, v2 - v1), half.Normal()) > 0;
}

using PolyFace = std::pair<std::vector<Vertex>, std::vector<std::size_t>>;

std::vector<PolyFace> FindPolyhedron(const std::vector<HalfSpace>& halves) {
  VertexMap points;

  // Find all combinations of the input planes to intersect.
  // This will give us all possible points.
  for (std::size_t i = 0; i < halves.size(); ++i) {
    const HalfSpace& half_i = halves[i];
    for (std::size_t j = i + 1; j < halves.size(); ++j) {
      const HalfSpace& half_j = halves[j];
      for (std::size_t k = i + 1; k < halves.size(); ++k) {
        const HalfSpace& half_k = halves[k];
        const auto vertex = ThreePlanes(half_i, half_j, half_k);
        if (vertex) {
          points.Insert(*vertex, std::set<size_t>({i, j, k}));
        }
      }
    }
  }

  // Throw out all the points in the map that are not in all the half-spaces.
  points.Filter([halves](const VertexMap::KeyVal& element) -> bool {
    const auto& [vertex, faces] = element;
    for (std::size_t i = 0; i < halves.size(); ++i) {
      // Don't consider the faces that contain the vertex.
      if (faces.find(i) == faces.end()) {
        const HalfSpace& h = halves[i];
        if (DotProduct(h.Normal(), vertex.ToVector()) + h.Position() >
            EPSILON) {
          // The vertex is not in the half-space h.
          // Discard the vertex
          return true;
        }
      }
    }
    return false;
  });

  // For each face, maintain a vector of all the edges on the face.
  // The edge is represented by a tuple (const Vertex*, const Vertex*,
  // other_face).
  // At this time, we don't try to order the vertices.
  // The Verticies are owned by the the map points.
  std::vector<
      std::vector<std::tuple<const Vertex*, const Vertex*, std::size_t>>>
      face_edges(halves.size());

  // Consider all pairs of vertices.
  for (size_t i = 0; i < points.size(); ++i) {
    const auto& [v_i, f_i] = points.get(i);
    for (size_t j = i + 1; j < points.size(); ++j) {
      const auto& [v_j, f_j] = points.get(j);
      // Count the number of faces in common.
      std::vector<std::size_t> faces;
      std::set_intersection(f_i.begin(), f_i.end(), f_j.begin(), f_j.end(),
                            std::back_inserter(faces));
      if (faces.size() == 2) {
        // Record the existence of an edge between v_i and v_j.
        face_edges[faces[0]].push_back(std::make_tuple(&v_i, &v_j, faces[1]));
        face_edges[faces[1]].push_back(std::make_tuple(&v_i, &v_j, faces[0]));
      }
    }
  }

  // Finally, we can build the list of vertices of each face, in
  // counterclockwise order, and the list of other adjacent faces
  // The two lists are the same size.
  std::vector<PolyFace> result;

  for (std::size_t i = 0; i < face_edges.size(); ++i) {
    std::vector<const Vertex*> vertices;
    std::vector<std::size_t> faces;

    const HalfSpace& half = halves[i];
    const auto& fe = face_edges[i];
    // Grab the first edge listed on the face.
    const auto& [v0, v1, f1] = fe[0];
    // Search for another edge containing v1.
    // When we find it, we will have three consecutive vertices.
    // Push them in counterclockwise order.
    for (std::size_t j = 1; j < fe.size(); ++j) {
      const auto& [v2, v3, f2] = fe[j];
      if (v2 == v1) {
        if (IsCounterclockwise(*v0, *v1, *v3, half)) {
          vertices.push_back(v0);
          vertices.push_back(v1);
          vertices.push_back(v3);
          faces.push_back(f1);
          faces.push_back(f2);
        } else {
          vertices.push_back(v3);
          vertices.push_back(v1);
          vertices.push_back(v0);
          faces.push_back(f2);
          faces.push_back(f1);
        }
        break;
      }
      if (v3 == v1) {
        if (IsCounterclockwise(*v0, *v1, *v2, half)) {
          vertices.push_back(v0);
          vertices.push_back(v1);
          vertices.push_back(v2);
          faces.push_back(f1);
          faces.push_back(f2);
        } else {
          vertices.push_back(v2);
          vertices.push_back(v1);
          vertices.push_back(v0);
          faces.push_back(f2);
          faces.push_back(f1);
        }
        break;
      }
    }
    if (vertices.empty()) {
      // We couldn't find another edge containing v1.
      // This can happen if the face is open.
      throw std::invalid_argument("Incomplete face");
    }

    for (;;) {
      // Find a vertex to add onto the end of vertices.
      const Vertex* next_v = nullptr;
      for (const auto& [v0, v1, f] : fe) {
        const Vertex* const last = vertices.back();
        const Vertex* const last2 = vertices[vertices.size() - 2];
        if (v0 == last) {
          if (v1 == last2) {
            // Don't pick the edge from the previous iteration.
            continue;
          }
          next_v = v1;
          faces.push_back(f);
          break;
        }
        if (v1 == last) {
          if (v0 == last2) {
            // Don't pick the edge from the previous iteration.
            continue;
          }
          next_v = v0;
          faces.push_back(f);
          break;
        }
      }
      if (next_v == nullptr) {
        // We couldn't find another edge containing last.
        // This can happen if the face is open.
        throw std::invalid_argument("Incomplete face");
      }
      if (next_v == vertices.front()) {
        // We have completed the loop. We're done.
        break;
      }
      if (vertices.size() >= fe.size()) {
        // Something went wrong. If the edges form a proper loop,
        // we cannot have more vertices the edges.
        throw std::logic_error("Loop in face?");
      }
      vertices.push_back(next_v);
    }

    // Add the vertices and faces we just computed to the result.
    // However, we cannot return the pointers in vertices, since
    // the Vertex objects the point to will not survive the function.
    // So we have to copy all the Vertex objects pointed to.
    result.emplace_back(std::vector<Vertex>(), faces);
    std::vector<Vertex>& tail = result.back().first;
    tail.reserve(vertices.size());
    for (const Vertex* v : vertices) {
      tail.push_back(*v);
    }
  }
  return result;
}

HalfSpace make_octa_face(const Vertex& p0, const Vertex& p1, const Vertex& p2) {
  const Vector c = CrossProduct(p2 - p1, p0 - p1);
  return HalfSpace(c.dx(), c.dy(), c.dz(), -DotProduct(p0.ToVector(), c));
}

// Test code for octahedron.
void test() {
  const Vertex NW(-1, 1, 0);
  const Vertex NE(1, 1, 0);
  const Vertex SW(-1, -1, 0);
  const Vertex SE(1, -1, 0);
  const Vertex TOP = Vertex(0, 0, std::sqrt(2.0));
  const Vertex BOTTOM = Vertex(0, 0, -std::sqrt(2.0));
  const auto f0 = make_octa_face(SE, NE, TOP);
  const auto f1 = make_octa_face(NE, NW, TOP);
  const auto f2 = make_octa_face(NW, SW, TOP);
  const auto f3 = make_octa_face(SW, SE, TOP);
  const auto f4 = make_octa_face(NE, SE, BOTTOM);
  const auto f5 = make_octa_face(NW, NE, BOTTOM);
  const auto f6 = make_octa_face(SW, NW, BOTTOM);
  const auto f7 = make_octa_face(SE, SW, BOTTOM);
  const auto poly = FindPolyhedron({f0, f1, f2, f3, f4, f5, f6, f7});
  for (std::size_t i = 0; i < poly.size(); ++i) {
    std::cout << "========= Face " << i << "\n";
    const auto& [vertices, faces] = poly[i];
    assert(vertices.size() == faces.size());
    for (std::size_t j = 0; j < vertices.size(); ++j) {
      std::cout << vertices[j].Format() << " " << faces[j] << "\n";
    }
  }
}

int main() { test(); }
