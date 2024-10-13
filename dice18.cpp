// Program to design 18-sided dice

// Make sure M_PI gets defined
#define _USE_MATH_DEFINES

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

static constexpr double EPSILON = 1.0e-10;

// A vector in 3-space.
class Vector {
 public:
  Vector(double dx, double dy, double dz) : dx_(dx), dy_(dy), dz_(dz) {}

  double dx() const { return dx_; }
  double dy() const { return dy_; }
  double dz() const { return dz_; }

  friend Vector operator-(const Vector& lhs, const Vector& rhs) {
    return Vector(lhs.dx_ - rhs.dx_, lhs.dy_ - rhs.dy_, lhs.dz_ - rhs.dz_);
  }

  double Magnitude() const {
    return std::sqrt(dx_ * dx_ + dy_ * dy_ + dz_ * dz_);
  }

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

// An infinite plane separates 3-space into two half spaces.
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

  // Writes a binary representation the data to a file.
  void Write(std::ofstream& out_file) const {
    out_file.write(reinterpret_cast<const char*>(&a_), sizeof(a_));
    out_file.write(reinterpret_cast<const char*>(&b_), sizeof(b_));
    out_file.write(reinterpret_cast<const char*>(&c_), sizeof(c_));
    out_file.write(reinterpret_cast<const char*>(&d_), sizeof(d_));
  }

 private:
  double a_;
  double b_;
  double c_;
  double d_;
};

// Returns true if a and b are within a floating point roundoff error
// from each other. Borrowed from Python. The tolerance values have
// been chosen experimentally.
bool IsClose(double a, double b) {
  return std::abs(a - b) <=
         std::max(1.0e-08 * std::max(std::abs(a), std::abs(b)), 1.5e-14);
}

// A Vertex is defined by its Cartesian coordinates in 3-space.
class Vertex {
 public:
  Vertex(double x, double y, double z) : x_(x), y_(y), z_(z) {}

  bool IsClose(const Vertex& other) const {
    return ::IsClose(x_, other.x_) && ::IsClose(y_, other.y_) &&
           ::IsClose(z_, other.z_);
  }

  friend bool operator==(const Vertex& lhs, const Vertex& rhs) {
    return lhs.x_ == rhs.x_ && lhs.y_ == rhs.y_ && lhs.z_ == rhs.z_;
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

// Return the point (if any) where three planes intersect.
std::optional<Vertex> ThreePlanes(const HalfSpace& p0, const HalfSpace& p1,
                                  const HalfSpace& p2) {
  const auto& intersection = PlaneIntersection(p0, p1);
  if (!intersection.has_value()) {
    // p0 and p1 are parallel.
    return std::nullopt;
  }

  // Find the point where the where the p0/p1 intersection crosses p2.
  const auto& [vertex, vector] = *intersection;
  const double denom = DotProduct(p2.Normal(), vector);
  if (std::abs(denom) < EPSILON) {
    // p2 and the intersection intersection are (almost) parallel;
    // there is no point to return.
    return std::nullopt;
  }
  const double t =
      (DotProduct(p2.Normal(), vertex.ToVector()) + p2.Position()) / denom;
  return Vertex(vertex.x() - t * vector.dx(), vertex.y() - t * vector.dy(),
                vertex.z() - t * vector.dz());
}

// For each vertex, records all the faces that touch it.
class VertexMap {
 public:
  using KeyVal = std::pair<Vertex, std::set<std::size_t>>;

  // The number of items in the map.
  std::size_t size() const { return map_.size(); }

  // Return an integer that can be used with get()
  size_t key(const Vertex& v) {
    return std::find_if(map_.begin(), map_.end(),
                        [&v](const KeyVal kv) { return kv.first == v; }) -
           map_.begin();
  }
  const KeyVal& get(std::size_t i) const { return map_[i]; }

  // Merges faces into the set of faces associated with v.
  // Use a relaxed compare to deal with roundoff errors when computing
  // the coordinates of vertices.
  void Insert(const Vertex& v, const std::set<std::size_t>& faces);

  // Removes every map entry for which filter(key_val) returns true.
  void Filter(std::function<bool(const KeyVal&)> filter) {
    map_.erase(std::remove_if(map_.begin(), map_.end(), filter), map_.end());
  }

 private:
  std::vector<KeyVal> map_;
};

void VertexMap::Insert(const Vertex& v, const std::set<std::size_t>& faces) {
  std::size_t i = 0;
  for (auto& [key, value] : map_) {
    if (key.IsClose(v)) {
      value.insert(faces.begin(), faces.end());
      return;
    }
    ++i;
  }
  map_.emplace_back(v, faces);
}

// Given three vertices v0, v1, v2 of a polygon in the border plane for half,
// returns whether they are in counterclockwise order, as viewed from outside
// the half-space.
bool IsCounterclockwise(const Vertex& v0, const Vertex& v1, const Vertex& v2,
                        const HalfSpace half) {
  return DotProduct(CrossProduct(v1 - v0, v2 - v1), half.Normal()) > 0;
}

// Computes the area of a convex polygon by dividing it into triangles and
// adding their areas. This only works if the input verticies are coplanar.
double Area(const std::vector<const Vertex*>& vertices) {
  assert(vertices.size() >= 2);
  const Vertex* const v0 = vertices.front();
  std::vector<Vector> diagonals;
  diagonals.reserve(vertices.size() - 1);
  for (std::size_t i = 1; i < vertices.size(); ++i) {
    diagonals.push_back(*vertices[i] - *v0);
  }
  double result = 0;
  for (std::size_t i = 1; i < diagonals.size(); ++i) {
    result += CrossProduct(diagonals[i - 1], diagonals[i]).Magnitude();
  }
  return 0.5 * result;
}

// Writes half-space parameters to a file, so that they can be
// processed by a different program.
void DumpHalves(const std::vector<HalfSpace>& halves,
                const std::string& filename) {
  std::cout << "Writing model to " << filename << "\n";
  std::ofstream out_file(filename, std::ios::binary);
  for (const HalfSpace& h : halves) {
    std::cout << h.Format() << "\n";
    h.Write(out_file);
  }
  out_file.close();
}

// Reads half-space parameters from a file. Used for debugging.
std::vector<HalfSpace> ReadHalves(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cout << std::format("Cannot read {}\n", filename);
    exit(1);
  }
  std::vector<HalfSpace> halves;
  std::vector<double> buffer;
  buffer.reserve(4);
  while (file.peek() != std::char_traits<char>::eof()) {
    double h;
    file.read(reinterpret_cast<char*>(&h), sizeof(h));
    buffer.push_back(h);
    if (buffer.size() == 4) {
      halves.emplace_back(buffer[0], buffer[1], buffer[2], buffer[3]);
      buffer.clear();
    }
  }
  file.close();
  assert(buffer.empty());
  return halves;
}

// The goal of this project is to try to find polyhedra whose faces
// are close in area. This function evaluates a set of half-spaces.
// It constructs the polyhedron and scores it.
// Scores are improved by having the faces be close to the same size.
// Perfect score is 1.0; worse scores are smaller. If the half spaces do
// not form a finite polyhedron, the score returned is -1.0.
double MeasurePolyhedron(const std::vector<HalfSpace>& halves) {
  VertexMap points;

  // Find all combinations of the input planes to intersect.
  // This will give us all possible points.
  for (std::size_t i = 0; i < halves.size(); ++i) {
    const HalfSpace& half_i = halves[i];
    for (std::size_t j = i + 1; j < halves.size(); ++j) {
      const HalfSpace& half_j = halves[j];
      for (std::size_t k = j + 1; k < halves.size(); ++k) {
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
  // We don't try to order the vertices. The verticies are owned by the
  // VertexMap points.
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
  // counterclockwise order.

  // Used to score the polyhedron.
  double min_area = std::numeric_limits<double>::max();
  double max_area = std::numeric_limits<double>::lowest();

  for (std::size_t i = 0; i < face_edges.size(); ++i) {
    std::vector<const Vertex*> vertices;

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
        } else {
          vertices.push_back(v3);
          vertices.push_back(v1);
          vertices.push_back(v0);
        }
        break;
      }
      if (v3 == v1) {
        if (IsCounterclockwise(*v0, *v1, *v2, half)) {
          vertices.push_back(v0);
          vertices.push_back(v1);
          vertices.push_back(v2);
        } else {
          vertices.push_back(v2);
          vertices.push_back(v1);
          vertices.push_back(v0);
        }
        break;
      }
    }
    if (vertices.empty()) {
      // We couldn't find another edge containing v1.
      // This can happen if the face is open.
      return -1.0;
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
          break;
        }
        if (v1 == last) {
          if (v0 == last2) {
            // Don't pick the edge from the previous iteration.
            continue;
          }
          next_v = v0;
          break;
        }
      }
      if (next_v == nullptr) {
        // We couldn't find another edge containing last.
        // This can happen if the face is open.
        return -1;
      }
      if (next_v == vertices.front()) {
        // We have completed the loop. We're done.
        break;
      }
      if (vertices.size() >= fe.size()) {
        // Something went wrong. If the edges form a proper loop,
        // we cannot have more vertices the edges.
        // In the past this problem is due to floating point roundoff
        // error, and was fixed by relaxing IsClose.
        std::cout << "Bad model?\n";
        DumpHalves(halves, "c:/users/sigma/documents/bad_model.bin");
        return -1;
      }
      vertices.push_back(next_v);
    }

    const double area = Area(vertices);
    if (area > max_area) {
      max_area = area;
    }
    if (area < min_area) {
      min_area = area;
    }
  }

  return min_area / max_area;
}

// Generates random polyhedra and scores them, keeping the best one.
class PolyFinder {
 public:
  PolyFinder() {}

  // Generates and scores as many random polyhedra as it can in
  // the given number of minutes. We expect this function to be
  // called only once.
  void Run(int minutes);

  // These values are available when Run returns.
  const std::vector<HalfSpace>& best() { return best_; }
  double best_score() const { return best_score_; }
  std::size_t success() const { return success_; }

 private:
  // Returns a random double r, where 0.0 < r < 1.0
  double RandFraction() {
    for (;;) {
      double result = dis_(gen_);
      if (0.0 < result && result < 1.0) {
        return result;
      }
    }
  }

  // Generates 18 half-spaces by choosing 9 random points on a sphere,
  // and for each point, including both the point and its antipode.
  std::vector<HalfSpace> GenHalves();

  std::random_device rd_;
  std::mt19937 gen_{rd_()};
  std::uniform_real_distribution<> dis_{0.0, 1.0};
  std::vector<HalfSpace> best_;
  double best_score_{-1.0};
  std::size_t success_{0};
};

std::vector<HalfSpace> PolyFinder::GenHalves() {
  std::vector<HalfSpace> result;
  result.reserve(18);
  while (result.size() < 18) {
    // We try to avoid points that are too close. This leads to
    // dihedral angles near 180 degrees and wonky floating
    // point calculations when computing their intersections.
    bool good_found = false;
    double x;
    double y;
    double z;
    // Try 25 times for a new point far enough away from all the others.
    for (int i = 0; i < 25; i++) {
      // Generate a random point on the unit sphere.
      // See https://mathworld.wolfram.com/SpherePointPicking.html
      const double theta = 2 * M_PI * RandFraction();
      const double phi = std::acos(2 * RandFraction() - 1);
      const double sin_phi = std::sin(phi);
      x = std::cos(theta) * sin_phi;
      y = std::sin(theta) * sin_phi;
      z = std::cos(phi);
      const Vector v(x, y, z);

      // Check if the new point is too close to any existing points.
      bool too_close = false;
      for (const HalfSpace& h : result) {
        if ((v - h.Normal()).Magnitude() < 0.35) {
          too_close = true;
          break;
        }
      }
      if (!too_close) {
        good_found = true;
        break;
      }
    }
    if (good_found) {
      // Push the half spaces defined by the point and its antipode.
      result.push_back(HalfSpace(x, y, z, -1.0));
      result.push_back(HalfSpace(-x, -y, -z, -1.0));
    } else {
      // There is no room for the next point.
      // Start over.
      std::cout << "Starting Over\n";
      result.clear();
    }
  }
  return result;
}

void PolyFinder::Run(int minutes) {
  std::size_t success = 0;
  std::size_t fail = 0;
  double best_score = -1.0;
  std::vector<HalfSpace> best;
  const std::chrono::time_point stop_time =
      std::chrono::steady_clock::now() + std::chrono::minutes(minutes);
  do {
    const std::vector<HalfSpace> h = GenHalves();
    const double score = MeasurePolyhedron(h);
    if (score >= 0) {
      ++success;
      if (score > best_score) {
        best_score = score;
        best = h;
      }
    } else {
      ++fail;
    }
  } while (std::chrono::steady_clock::now() < stop_time);

  // Report the results.
  success_ = success;
  best_score_ = best_score;
  best_ = best;
}

int main() {
  const unsigned n = std::thread::hardware_concurrency();
  std::cout << "Running " << n << " threads\n";
  std::unique_ptr<PolyFinder[]> finders = std::make_unique<PolyFinder[]>(n);
  std::vector<std::thread> threads;
  threads.reserve(n);

  // Start all the threads. Each will run for six hours.
  for (unsigned i = 0; i < n; ++i) {
    threads.push_back(std::thread([&finders, i]() { finders[i].Run(6*60); }));
  }

  // Wait for them all to finish.
  double best_score = -1.0;
  std::size_t num_candidates = 0;
  unsigned best_thread = n;  // Invalid value
  for (unsigned i = 0; i < n; ++i) {
    threads[i].join();
    num_candidates += finders[i].success();
    const double b = finders[i].best_score();
    if (b > 0 && b > best_score) {
      best_score = b;
      best_thread = i;
    }
  }
  if (num_candidates == 0) {
    std::cout << "Didn't find any candidates?\n";
    return 0;
  }

  // Write the best result into a file.
  std::cout << "There were " << num_candidates << " candidates\n";
  std::cout << "Best score " << best_score << "\n";
  DumpHalves(finders[best_thread].best(),
             "c:/users/sigma/documents/best_halves.bin");
  return 0;
}
