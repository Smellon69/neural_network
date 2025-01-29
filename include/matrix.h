#ifndef MY_NEURAL_NET_MATRIX_H_
#define MY_NEURAL_NET_MATRIX_H_

#include <cstddef>
#include <vector>
#include <functional>

/**
 * @file matrix.h
 * @brief Defines a Matrix class with parallelized operations.
 */

namespace nn {

	/**
	 * @class Matrix
	 * @brief Encapsulates a 2D matrix with row-major storage and
	 *        provides parallel operations (add, multiply, etc.).
	 */
	class Matrix {
	public:
		/**
		 * @brief Constructs a matrix with specified rows, cols.
		 * @param rows Number of rows
		 * @param cols Number of columns
		 * @param randomize If true, randomize matrix values
		 */
		Matrix(size_t rows, size_t cols, bool randomize = false);

		/**
		 * @brief Default constructor for empty matrix placeholders.
		 */
		Matrix();

		/**
		 * @brief Element access (mutable).
		 * @param r Row index
		 * @param c Column index
		 * @return Reference to element
		 */
		double& operator()(size_t r, size_t c);

		/**
		 * @brief Element access (const).
		 * @param r Row index
		 * @param c Column index
		 * @return Const value of element
		 */
		double operator()(size_t r, size_t c) const;

		/**
		 * @return Number of rows in matrix.
		 */
		size_t rows() const;

		/**
		 * @return Number of columns in matrix.
		 */
		size_t cols() const;

		/**
		 * @brief Parallel matrix multiplication: C = A * B.
		 * @param A Left operand
		 * @param B Right operand
		 * @return Result of A*B
		 */
		static Matrix multiply(const Matrix& A, const Matrix& B);

		/**
		 * @brief Parallel element-wise add: C = A + B.
		 * @param A Left operand
		 * @param B Right operand
		 * @return Result of A+B
		 */
		static Matrix add(const Matrix& A, const Matrix& B);

		/**
		 * @brief In-place transform using a unary function.
		 * @param func Function to apply to each element
		 */
		void applyFunction(const std::function<double(double)>& func);

		/**
		 * @brief Transpose the given matrix.
		 * @param M Matrix to transpose
		 * @return Transposed matrix
		 */
		static Matrix transpose(const Matrix& M);

		/**
		 * @return Reference to underlying data vector.
		 */
		std::vector<double>& data();

		/**
		 * @return Const reference to underlying data vector.
		 */
		const std::vector<double>& data() const;

	private:
		size_t m_rows;
		size_t m_cols;
		std::vector<double> m_data;

		/**
		 * @brief Helper to random-initialize data.
		 */
		void randomInit();
	};

}  // namespace nn

#endif  // MY_NEURAL_NET_MATRIX_H_
