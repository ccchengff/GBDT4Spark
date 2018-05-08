package org.dma.gbdt4spark.util;

import java.util.List;
import java.util.Random;

public class Maths {
    public static final float EPSILON = 1e-8f;

    public static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    public static double sigmoid(double x) {
        return (1.0 / (1.0 + Math.exp(-x)));
    }

    public static int sqr(int x) {
        return x * x;
    }

    public static float sqr(float x) {
        return x * x;
    }

    public static double sqr(double x) {
        return x * x;
    }

    public static void softmax(double[] rec) {
        double wmax = rec[0];
        for (int i = 1; i < rec.length; ++i) {
            wmax = Math.max(rec[i], wmax);
        }
        double wsum = 0.0;
        for (int i = 0; i < rec.length; ++i) {
            rec[i] = Math.exp(rec[i] - wmax);
            wsum += rec[i];
        }
        for (int i = 0; i < rec.length; ++i) {
            rec[i] /= wsum;
        }
    }

    public static void softmax(float[] rec) {
        float wmax = rec[0];
        for (int i = 1; i < rec.length; ++i) {
            wmax = Math.max(rec[i], wmax);
        }
        float wsum = 0.0f;
        for (int i = 0; i < rec.length; ++i) {
            rec[i] = (float) Math.exp(rec[i] - wmax);
            wsum += rec[i];
        }
        for (int i = 0; i < rec.length; ++i) {
            rec[i] /= wsum;
        }
    }

    public static double thresholdL1(double w, double lambda) {
        if (w > +lambda)
            return w - lambda;
        if (w < -lambda)
            return w + lambda;
        return 0.0;
    }

    public static float thresholdL1(float w, float lambda) {
        if (w > +lambda)
            return w - lambda;
        if (w < -lambda)
            return w + lambda;
        return 0.0f;
    }

    public static boolean isEven(int v) {
        return v % 2 == 0;
    }

    public static boolean areZeros(float[] floats) {
        for (float f : floats) {
            if (Math.abs(f) > EPSILON)
                return false;
        }
        return true;
    }

    public static int argmax(float[] floats) {
        int res = 0;
        float max = floats[res];
        for (int i = 1; i < floats.length; i++) {
            if (floats[i] > max) {
                res = i;
                max = floats[i];
            }
        }
        return res;
    }

    public static int parent(int nodeId) {
        return (nodeId - 1) / 2;
    }

    public static int sibling(int nodeId) {
        if (isEven(nodeId))
            return nodeId - 1;
        else
            return nodeId + 1;
    }

    public static int pow(int a, int b) {
        if (b == 0)
            return 1;
        if (b == 1)
            return a;
        if (isEven(b))
            return pow(a * a, b / 2); // even a=(a^2)^b/2
        else
            return a * pow(a * a, b / 2); // odd a=a*(a^2)^b/2
    }

    public static float[] unique(float[] array) {
        int cnt = 1;
        for (int i = 1; i < array.length; i++) {
            if (array[i] != array[i - 1])
                cnt++;
        }
        if (cnt != array.length) {
            float[] res = new float[cnt];
            res[0] = array[0];
            int index = 1;
            for (int i = 1; i < array.length; i++) {
                if (array[i] != array[i - 1])
                    res[index++] = array[i];
            }
            return res;
        } else {
            return array;
        }
    }

    public static void shuffle(int[] array) {
        int index, temp;
        Random random = new Random();
        for (int i = array.length - 1; i > 0; i--) {
            index = random.nextInt(i + 1);
            temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }

    public static float[] floatListToArray(List<Float> list) {
        int size = list.size();
        float[] arr = new float[size];
        for (int i = 0; i < size; i++)
            arr[i] = list.get(i);
        return arr;
    }

    public static int indexOf(float[] splits, float x) {
        int l = 0, r = splits.length - 1;
        while (l <= r) {
            int mid = (l + r) >> 1;
            if (splits[mid] <= x) {
                if (mid + 1 == splits.length || splits[mid + 1] > x)
                    return mid;
                else
                    l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return Math.max(0, Math.min(splits.length - 1, (l + r) >> 1)); // should never reach here
    }

    public static int indexOf(double[] splits, double x) {
        int l = 0, r = splits.length - 1;
        while (l <= r) {
            int mid = (l + r) >> 1;
            if (splits[mid] <= x) {
                if (mid + 1 == splits.length || splits[mid + 1] > x)
                    return mid;
                else
                    l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return Math.max(0, Math.min(splits.length - 1, (l + r) >> 1)); // should never reach here
    }

    public static float dot(float[] a, float[] b) {
        int dim = Math.min(a.length, b.length);
        float res = 0.0f;
        for (int i = 0; i < dim; i++)
            res += a[i] * b[i];
        return res;
    }

    public static int indexOfLowerTriangularMatrix(int row, int col) {
        return ((row * (row + 1)) >> 1) + col;
    }

    public static int indexOfUpperTriangularMatrix(int row, int col, int n) {
        return row * (2 * n - row + 1) / 2 + col;
    }

    /**
     * Compute matrix M = L*L(T), where L is a lower triangular matrix
     *
     * @param L a lower triangular matrix
     * @param n dimension
     * @return matrix L*(L^T)
     */
    public static float[] LLT(float[] L, int n) {
        float[] M = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float s = 0.0f;
                int rowI = indexOfLowerTriangularMatrix(i, 0);
                int colJ = indexOfLowerTriangularMatrix(j, 0);
                for (int k = 0; k < i + 1; k++) {
                    float Lik = k <= i ? L[rowI + k] : 0.0f;
                    float LTjk = k <= j ? L[colJ + k] : 0.0f;
                    s += Lik * LTjk;
                }
                M[i * n + j] = s;
            }
        }
        return M;
    }

    /**
     * Matrix-vector multiplication of lower triangular matrix and vector
     *
     * @param L a lower triangular matrix
     * @param b a vector
     * @param n dimension
     * @return vector L*b
     */
    public static float[] Lb(float[] L, float[] b, int n) {
        float[] res = new float[n];
        for (int i = 0; i < n; i++) {
            int rowI = indexOfLowerTriangularMatrix(i, 0);
            float s = 0.0f;
            for (int j = 0; j < i + 1; j++) {
                s += L[rowI + j] * b[j];
            }
            res[i] = s;
        }
        return res;
    }

    /**
     * Matrix-vector multiplication of transposition of lower triangular matrix and vector
     *
     * @param L a lower triangular matrix
     * @param b a vector
     * @param n dimension
     * @return vector (L^T)*b
     */
    public static float[] LTb(float[] L, float[] b, int n) {
        float[] res = new float[n];
        for (int i = 0; i < n; i++) {
            int rowI = indexOfLowerTriangularMatrix(i, 0);
            for (int j = 0; j < i + 1; j++) {
                res[j] += L[rowI + j] * b[i];
            }
        }
        return res;
    }

    /**
     * Forward substitution to solve Ly = b
     *
     * @param L a lower triangular matrix
     * @param b a vector
     * @param n dimension
     * @return vector y
     */
    public static float[] forwardSubstitution(float[] L, float[] b, int n) {
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            float s = 0.0f;
            int rowI = indexOfLowerTriangularMatrix(i, 0);
            for (int j = 0; j < i; j++) {
                s += L[rowI + j] * y[j];
            }
            y[i] = (b[i] - s) / L[rowI + i];
        }
        return y;
    }


    /**
     * Backward substitution to solve Ux = y
     *
     * @param U an upper triangular matrix
     * @param y a vector
     * @param n dimension
     * @return vector x
     */
    public static float[] backwardSubstitution(float[] U, float[] y, int n) {
        float[] x = new float[n];
        for (int i = n - 1; i >= 0; i--) {
            float s = 0.0f;
            int rowI = indexOfUpperTriangularMatrix(i, 0, n);
            for (int j = n - 1; j > i; j--) {
                s += U[rowI + j] * x[j];
            }
            x[i] = (y[i] - s) / U[rowI + i];
        }
        return x;
    }

    /**
     * Backward substitution to solve Ux = y, but given L = U^T
     *
     * @param L a lower triangular matrix
     * @param y a vector
     * @param n dimension
     * @return vector x
     */
    public static float[] backwardSubstitutionL(float[] L, float[] y, int n) {
        float[] x = new float[n];
        for (int i = n - 1; i >= 0; i--) {
            float s = 0.0f;
            for (int j = n - 1; j > i; j--) {
                int index = indexOfLowerTriangularMatrix(j, i);
                s += L[index] * x[j];
            }
            int index = indexOfLowerTriangularMatrix(i, i);
            x[i] = (y[i] - s) / L[index];
        }
        return x;
    }

    /**
     * Cholesky Decomposition of matrix A
     *
     * @param A a symmetric positive matrix, represented in lower triangular matrix
     * @param n dimension
     * @return lower triangular matrix L s.t. A = L*(L^T)
     */
    public static float[] choleskyDecomposition(float[] A, int n) {
        float[] L = new float[A.length];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i + 1; j++) {
                float s = 0;
                int rowI = indexOfLowerTriangularMatrix(i, 0);
                int rowJ = indexOfLowerTriangularMatrix(j, 0);
                for (int k = 0; k < j; k++) {
                    s += L[rowI + k] + L[rowJ + k];
                }
                L[rowI + j] = (i == j) ? (float) Math.sqrt(A[rowI + i] - s)
                        : 1.0f / L[rowJ + j] * (A[rowI + j] - s);
            }
        }
        return L;
    }

    /**
     * Solve linear system Ax = b with Cholesky Decomposition
     *
     * @param A a symmetric positive matrix, represented in lower triangular matrix
     * @param b a vector
     * @param n dimension
     * @return x = A^(-1)b
     */
    public static float[] solveLinearSystemWithCholeskyDecomposition(float[] A, float[] b, int n) {
        float[] L = choleskyDecomposition(A, n);
        float[] y = forwardSubstitution(L, b, n);
        float[] x = backwardSubstitutionL(L, y, n);
        return x;
    }
}
