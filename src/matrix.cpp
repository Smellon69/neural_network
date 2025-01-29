#include "../include/matrix.h"

#include <algorithm>
#include <cassert>
#include <execution>
#include <numeric>
#include <random>

namespace nn {

    Matrix::Matrix(size_t rows, size_t cols, bool randomize)
        : m_rows(rows), m_cols(cols), m_data(rows* cols, 0.0) {
        if (randomize) {
            randomInit();
        }
    }

    Matrix::Matrix() : m_rows(0), m_cols(0), m_data{} {}

    double& Matrix::operator()(size_t r, size_t c) {
        return m_data[r * m_cols + c];
    }

    double Matrix::operator()(size_t r, size_t c) const {
        return m_data[r * m_cols + c];
    }

    size_t Matrix::rows() const { return m_rows; }
    size_t Matrix::cols() const { return m_cols; }

    Matrix Matrix::multiply(const Matrix& A, const Matrix& B) {
        assert(A.cols() == B.rows() && "Incompatible matrix dimensions!");

        Matrix C(A.rows(), B.cols(), false);

        // Index range for all (i, j) in C
        std::vector<size_t> indices(A.rows() * B.cols());
        std::iota(indices.begin(), indices.end(), 0);

        auto multiplyCell = [&](size_t idx) {
            size_t i = idx / B.cols();
            size_t j = idx % B.cols();
            double sum = 0.0;
            for (size_t k = 0; k < A.cols(); ++k) {
                sum += A(i, k) * B(k, j);
            }
            return sum;
            };

        std::transform(std::execution::par,
            indices.begin(), indices.end(),
            C.m_data.begin(),
            multiplyCell);

        return C;
    }

    Matrix Matrix::add(const Matrix& A, const Matrix& B) {
        assert(A.rows() == B.rows() && A.cols() == B.cols());
        Matrix C(A.rows(), A.cols());
        std::transform(std::execution::par,
            A.m_data.begin(), A.m_data.end(),
            B.m_data.begin(),
            C.m_data.begin(),
            std::plus<double>());
        return C;
    }

    void Matrix::applyFunction(const std::function<double(double)>& func) {
        std::transform(std::execution::par,
            m_data.begin(), m_data.end(),
            m_data.begin(),
            func);
    }

    Matrix Matrix::transpose(const Matrix& M) {
        Matrix T(M.m_cols, M.m_rows);
        // Simple version (not parallel)
        for (size_t r = 0; r < M.m_rows; ++r) {
            for (size_t c = 0; c < M.m_cols; ++c) {
                T(c, r) = M(r, c);
            }
        }
        return T;
    }

    std::vector<double>& Matrix::data() {
        return m_data;
    }

    const std::vector<double>& Matrix::data() const {
        return m_data;
    }

    void Matrix::randomInit() {
        static std::mt19937 rng{ std::random_device{}() };
        static std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& val : m_data) {
            val = dist(rng);
        }
    }

}  // namespace nn
