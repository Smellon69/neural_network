/**
 * @file test_matrix.cpp
 * @brief Tests for the Matrix class using simple assert-based checks.
 */

#include <cassert>
#include <iostream>
#include "../include/matrix.h"

namespace test_matrix {

    using nn::Matrix;

    /**
     * @brief Tests basic initialization of Matrix.
     */
    static void testBasicInitialization() {
        Matrix m(3, 3);
        assert(m.rows() == 3 && "Matrix should have 3 rows");
        assert(m.cols() == 3 && "Matrix should have 3 cols");

        // By default, the elements are zero-initialized
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                assert(m(r, c) == 0.0 && "Matrix element should be 0");
            }
        }
    }

    /**
     * @brief Tests random initialization of Matrix.
     */
    static void testRandomInitialization() {
        // We can't strictly test "randomness", but at least ensure no crash.
        Matrix m(3, 3, true);
        // Check shape
        assert(m.rows() == 3 && "Matrix should have 3 rows");
        assert(m.cols() == 3 && "Matrix should have 3 cols");
    }

    /**
     * @brief Tests matrix multiply and add.
     */
    static void testMultiplyAdd() {
        Matrix A(2, 2);
        Matrix B(2, 2);

        // A = [[1, 2], [3, 4]]
        A(0, 0) = 1; A(0, 1) = 2;
        A(1, 0) = 3; A(1, 1) = 4;

        // B = [[5, 6], [7, 8]]
        B(0, 0) = 5; B(0, 1) = 6;
        B(1, 0) = 7; B(1, 1) = 8;

        // C = A * B = [[19, 22], [43, 50]]
        Matrix C = Matrix::multiply(A, B);
        assert(C(0, 0) == 19.0);
        assert(C(0, 1) == 22.0);
        assert(C(1, 0) == 43.0);
        assert(C(1, 1) == 50.0);

        // D = A + B = [[6, 8], [10, 12]]
        Matrix D = Matrix::add(A, B);
        assert(D(0, 0) == 6.0);
        assert(D(0, 1) == 8.0);
        assert(D(1, 0) == 10.0);
        assert(D(1, 1) == 12.0);
    }

    /**
     * @brief Runs all Matrix-related tests in sequence.
     */
    void runAllMatrixTests() {
        std::cout << "[test_matrix] Running tests...\n";
        testBasicInitialization();
        testRandomInitialization();
        testMultiplyAdd();
        std::cout << "[test_matrix] All tests passed!\n";
    }

}  // namespace test_matrix
