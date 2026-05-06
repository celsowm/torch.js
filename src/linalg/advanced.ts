/**
 * Advanced CPU-based linear algebra functions.
 * SVD, EIGH, QR, SOLVE, PINV
 */

import { Tensor } from '../tensor';
import { createBufferWithData } from '../backend';

// ============================================================================
// Helpers
// ============================================================================

async function readMatrixToCPU(t: Tensor): Promise<Float64Array> {
    const data = await t.toArray();
    return new Float64Array(data);
}

function createTensorFromCPU(data: Float64Array, shape: number[]): Tensor {
    const buffer = createBufferWithData(new Float32Array(data), 'float32');
    return new Tensor({ buffer, shape, dtype: 'float32', device: 'webgpu', requires_grad: false });
}

function copy2D(A: Float64Array[]): Float64Array[] {
    return A.map(row => new Float64Array(row));
}

function eye2D(n: number): Float64Array[] {
    const I: Float64Array[] = [];
    for (let i = 0; i < n; i++) {
        const row = new Float64Array(n);
        row[i] = 1;
        I.push(row);
    }
    return I;
}

function transpose2D(A: Float64Array[]): Float64Array[] {
    const m = A.length;
    const n = A[0].length;
    const T: Float64Array[] = [];
    for (let j = 0; j < n; j++) {
        T.push(new Float64Array(m));
        for (let i = 0; i < m; i++) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

function matmul2D(A: Float64Array[], B: Float64Array[]): Float64Array[] {
    const m = A.length;
    const k = A[0].length;
    const n = B[0].length;
    const C: Float64Array[] = [];
    for (let i = 0; i < m; i++) {
        C.push(new Float64Array(n));
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let p = 0; p < k; p++) sum += A[i][p] * B[p][j];
            C[i][j] = sum;
        }
    }
    return C;
}

function flattenMatrix(mat: Float64Array[]): Float64Array {
    const rows = mat.length;
    const cols = mat[0].length;
    const result = new Float64Array(rows * cols);
    for (let i = 0; i < rows; i++) result.set(mat[i], i * cols);
    return result;
}

// ============================================================================
// SVD via Jacobi rotations
// ============================================================================

function svd2D(A: Float64Array[], fullMatrices: boolean): { U: Float64Array[]; S: Float64Array; Vh: Float64Array[] } {
    const M = A.length;
    const N = A[0].length;
    const K = Math.min(M, N);

    let B = copy2D(A);
    let V = eye2D(N);
    const maxIter = 200;
    const tol = 1e-12;

    for (let iter = 0; iter < maxIter; iter++) {
        let maxOff = 0;
        for (let p = 0; p < M; p++) {
            for (let i = 0; i < K; i++) {
                for (let j = i + 1; j < N; j++) {
                    maxOff = Math.max(maxOff, Math.abs(B[p][i] * B[p][j]));
                }
            }
        }
        if (maxOff < tol) break;

        for (let i = 0; i < K; i++) {
            for (let j = i + 1; j < N; j++) {
                let aii = 0, ajj = 0, aij = 0;
                for (let p = 0; p < M; p++) {
                    aii += B[p][i] * B[p][i];
                    ajj += B[p][j] * B[p][j];
                    aij += B[p][i] * B[p][j];
                }
                if (Math.abs(aij) < 1e-15) continue;

                const tau = (ajj - aii) / (2 * aij);
                const sign = tau >= 0 ? 1 : -1;
                const t = sign / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
                const c = 1 / Math.sqrt(1 + t * t);
                const s = t * c;

                for (let p = 0; p < M; p++) {
                    const bi = B[p][i], bj = B[p][j];
                    B[p][i] = c * bi + s * bj;
                    B[p][j] = -s * bi + c * bj;
                }
                for (let p = 0; p < N; p++) {
                    const vi = V[p][i], vj = V[p][j];
                    V[p][i] = c * vi + s * vj;
                    V[p][j] = -s * vi + c * vj;
                }
            }
        }
    }

    // Singular values = column norms of B
    const S = new Float64Array(K);
    for (let i = 0; i < K; i++) {
        let norm = 0;
        for (let p = 0; p < M; p++) norm += B[p][i] * B[p][i];
        S[i] = Math.sqrt(norm);
    }

    // Sort descending
    const idx = Array.from({ length: K }, (_, i) => i);
    idx.sort((a, b) => S[b] - S[a]);
    const sortedS = new Float64Array(K);
    for (let i = 0; i < K; i++) sortedS[i] = S[idx[i]];

    // U[:,j] = A * V[:,j] / S[j]
    const Vt = transpose2D(V);
    const U: Float64Array[] = [];
    for (let i = 0; i < M; i++) {
        U.push(new Float64Array(K));
        for (let j = 0; j < K; j++) {
            if (sortedS[j] > 1e-15) {
                let sum = 0;
                for (let p = 0; p < N; p++) sum += A[i][p] * V[p][idx[j]];
                U[i][j] = sum / sortedS[j];
            }
        }
    }

    // Vh = V^T with sorted rows
    const Vh: Float64Array[] = [];
    for (let i = 0; i < K; i++) {
        Vh.push(new Float64Array(N));
        for (let j = 0; j < N; j++) Vh[i][j] = V[j][idx[i]];
    }

    return { U, S: sortedS, Vh };
}

/**
 * Singular Value Decomposition.
 * @pytorch torch.linalg.svd
 */
export const svd = async (input: Tensor, fullMatrices: boolean = true): Promise<{ U: Tensor; S: Tensor; Vh: Tensor }> => {
    const shape = input.shape;
    const ndim = shape.length;
    if (ndim < 2) throw new Error('linalg.svd: expected tensor with at least 2 dimensions');

    const M = shape[ndim - 2], N = shape[ndim - 1];
    const K = Math.min(M, N);
    const batch = ndim > 2 ? shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
    const batchShape = ndim > 2 ? shape.slice(0, -2) : [];

    const flatData = await readMatrixToCPU(input);
    const allU: Float64Array[][] = [], allS: Float64Array[] = [], allVh: Float64Array[][] = [];

    for (let b = 0; b < batch; b++) {
        const matData = flatData.slice(b * M * N, (b + 1) * M * N);
        const A: Float64Array[] = [];
        for (let i = 0; i < M; i++) A.push(new Float64Array(matData.slice(i * N, (i + 1) * N)));

        const { U, S, Vh } = svd2D(A, fullMatrices);
        allU.push(U);
        allS.push(S);
        allVh.push(Vh);
    }

    if (batch === 1) {
        return {
            U: createTensorFromCPU(flattenMatrix(allU[0]), [allU[0].length, allU[0][0].length]),
            S: createTensorFromCPU(allS[0], [K]),
            Vh: createTensorFromCPU(flattenMatrix(allVh[0]), [allVh[0].length, allVh[0][0].length]),
        };
    }

    const uCols = allU[0][0].length, vhCols = allVh[0][0].length;
    const Uflat = new Float64Array(batch * allU[0].length * uCols);
    const Sflat = new Float64Array(batch * K);
    const Vhflat = new Float64Array(batch * allVh[0].length * vhCols);

    for (let b = 0; b < batch; b++) {
        Uflat.set(flattenMatrix(allU[b]), b * allU[0].length * uCols);
        Sflat.set(allS[b], b * K);
        Vhflat.set(flattenMatrix(allVh[b]), b * allVh[0].length * vhCols);
    }

    return {
        U: createTensorFromCPU(Uflat, [...batchShape, allU[0].length, uCols]),
        S: createTensorFromCPU(Sflat, [...batchShape, K]),
        Vh: createTensorFromCPU(Vhflat, [...batchShape, allVh[0].length, vhCols]),
    };
};

// ============================================================================
// EIGH - symmetric eigenvalue decomposition via QR algorithm
// ============================================================================

function householderTridiagonal(A: Float64Array[]): { T: Float64Array[]; Q: Float64Array[] } {
    const n = A.length;
    const Q = eye2D(n);
    const T = copy2D(A);

    for (let k = 0; k < n - 2; k++) {
        const m = n - k - 1;
        const x = new Float64Array(m);
        for (let i = 0; i < m; i++) x[i] = T[k + 1 + i][k];

        const norm = Math.sqrt(x.reduce((s, v) => s + v * v, 0));
        if (norm < 1e-15) continue;

        const sign = x[0] >= 0 ? 1 : -1;
        x[0] += sign * norm;
        const vNorm = Math.sqrt(x.reduce((s, v) => s + v * v, 0));
        if (vNorm < 1e-15) continue;
        for (let i = 0; i < m; i++) x[i] /= vNorm;

        // vFull embedded in size n
        const vFull = new Float64Array(n);
        for (let i = 0; i < m; i++) vFull[k + 1 + i] = x[i];

        // T = (I - 2vv^T) * T * (I - 2vv^T)
        // Step 1: T = H * T
        const HT: Float64Array[] = [];
        for (let i = 0; i < n; i++) {
            HT.push(new Float64Array(n));
            for (let j = 0; j < n; j++) {
                let dot = 0;
                for (let p = 0; p < n; p++) dot += vFull[p] * T[p][j];
                HT[i][j] = T[i][j] - 2 * vFull[i] * dot;
            }
        }
        // Step 2: T = HT * H
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                let dot = 0;
                for (let p = 0; p < n; p++) dot += HT[i][p] * vFull[p];
                T[i][j] = HT[i][j] - 2 * dot * vFull[j];
            }
        }

        // Accumulate Q = Q * H
        const newQ: Float64Array[] = [];
        for (let i = 0; i < n; i++) {
            newQ.push(new Float64Array(n));
            for (let j = 0; j < n; j++) {
                let dot = 0;
                for (let p = 0; p < n; p++) dot += Q[i][p] * vFull[p];
                newQ[i][j] = Q[i][j] - 2 * dot * vFull[j];
            }
        }
        for (let i = 0; i < n; i++) Q[i] = newQ[i];
    }

    return { T, Q };
}

function qrIterationTridiagonal(diag: Float64Array, offdiag: Float64Array, n: number, maxIter = 1000, tol = 1e-12) {
    const d = new Float64Array(diag);
    const e = new Float64Array(offdiag);
    const Z = eye2D(n);

    let remaining = n;
    while (remaining > 1) {
        let iter = 0;
        while (iter < maxIter) {
            if (Math.abs(e[remaining - 2]) < tol * (Math.abs(d[remaining - 1]) + Math.abs(d[remaining - 2]) + 1e-15)) break;

            // Wilkinson shift
            const a = d[remaining - 2], b = e[remaining - 2], c = d[remaining - 1];
            const delta = (a - c) / 2;
            const mu = c - Math.sign(delta) * b * b / (Math.abs(delta) + Math.sqrt(delta * delta + b * b));

            // Implicit QR step — start from active subblock
            const start = n - remaining;
            let g = d[start] - mu;
            for (let i = start; i < n - 1; i++) {
                const r = Math.sqrt(g * g + e[i] * e[i]);
                const cs = g / r, sn = e[i] / r;
                if (i > start) e[i - 1] = r;
                g = cs * (d[i] - mu) + sn * e[i];
                d[i] = cs * g + sn * (cs * e[i] - sn * (d[i] - mu));
                const tmp = cs * e[i] - sn * (d[i] - mu);
                if (i < n - 2) {
                    e[i] = cs * tmp + sn * d[i + 1];
                    d[i + 1] = -sn * tmp + cs * d[i + 1];
                } else {
                    e[i] = tmp;
                }
                // Accumulate
                for (let k = 0; k < n; k++) {
                    const zi = Z[k][i], zj = Z[k][i + 1];
                    Z[k][i] = cs * zi + sn * zj;
                    Z[k][i + 1] = -sn * zi + cs * zj;
                }
            }
            iter++;
        }
        remaining--;
    }

    // Sort ascending
    const idx = Array.from({ length: n }, (_, i) => i);
    idx.sort((a, b) => d[a] - d[b]);

    const eigenvalues = new Float64Array(n);
    const eigenvectors: Float64Array[] = [];
    for (let i = 0; i < n; i++) eigenvalues[i] = d[idx[i]];
    // Create eigenvector matrix
    for (let i = 0; i < n; i++) eigenvectors.push(new Float64Array(n));
    for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) eigenvectors[i][j] = Z[i][idx[j]];
    }

    return { eigenvalues, eigenvectors };
}

/**
 * Eigenvalue decomposition for symmetric/Hermitian matrices.
 * @pytorch torch.linalg.eigh
 */
export const eigh = async (input: Tensor, UPLO: 'L' | 'U' = 'L'): Promise<{ eigenvalues: Tensor; eigenvectors: Tensor }> => {
    const shape = input.shape;
    const ndim = shape.length;
    if (ndim < 2) throw new Error('linalg.eigh: expected tensor with at least 2 dimensions');

    const N = shape[ndim - 1], M = shape[ndim - 2];
    if (N !== M) throw new Error('linalg.eigh: expected square matrix');

    const batch = ndim > 2 ? shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
    const batchShape = ndim > 2 ? shape.slice(0, -2) : [];

    const flatData = await readMatrixToCPU(input);
    const allEvals: Float64Array[] = [], allEvecs: Float64Array[][] = [];

    for (let b = 0; b < batch; b++) {
        const matData = flatData.slice(b * N * N, (b + 1) * N * N);
        const A: Float64Array[] = [];
        for (let i = 0; i < N; i++) A.push(new Float64Array(matData.slice(i * N, (i + 1) * N)));

        // Symmetrize
        const At = transpose2D(A);
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) A[i][j] = (A[i][j] + At[i][j]) / 2;

        const { T } = householderTridiagonal(A);
        const diag = new Float64Array(N);
        const offdiag = new Float64Array(N - 1);
        for (let i = 0; i < N; i++) diag[i] = T[i][i];
        for (let i = 0; i < N - 1; i++) offdiag[i] = T[i][i + 1];

        const { eigenvalues, eigenvectors } = qrIterationTridiagonal(diag, offdiag, N);
        allEvals.push(eigenvalues);
        allEvecs.push(eigenvectors);
    }

    if (batch === 1) {
        return {
            eigenvalues: createTensorFromCPU(allEvals[0], [N]),
            eigenvectors: createTensorFromCPU(flattenMatrix(allEvecs[0]), [N, N]),
        };
    }

    const evalFlat = new Float64Array(batch * N);
    const evecFlat = new Float64Array(batch * N * N);
    for (let b = 0; b < batch; b++) {
        evalFlat.set(allEvals[b], b * N);
        evecFlat.set(flattenMatrix(allEvecs[b]), b * N * N);
    }

    return {
        eigenvalues: createTensorFromCPU(evalFlat, [...batchShape, N]),
        eigenvectors: createTensorFromCPU(evecFlat, [...batchShape, N, N]),
    };
};

// ============================================================================
// QR via Householder reflections
// ============================================================================

function qr2D(A: Float64Array[], mode: 'reduced' | 'complete'): { Q: Float64Array[]; R: Float64Array[] } {
    const M = A.length, N = A[0].length;
    const K = Math.min(M, N);
    const R = copy2D(A);
    const Qfull = eye2D(M);

    for (let j = 0; j < K; j++) {
        const m = M - j;
        const x = new Float64Array(m);
        for (let i = 0; i < m; i++) x[i] = R[j + i][j];

        const norm = Math.sqrt(x.reduce((s, v) => s + v * v, 0));
        if (norm < 1e-15) continue;

        const sign = x[0] >= 0 ? 1 : -1;
        x[0] += sign * norm;
        const vNorm = Math.sqrt(x.reduce((s, v) => s + v * v, 0));
        if (vNorm < 1e-15) continue;
        for (let i = 0; i < m; i++) x[i] /= vNorm;

        // Apply H to R
        for (let col = j; col < N; col++) {
            let dot = 0;
            for (let i = 0; i < m; i++) dot += x[i] * R[j + i][col];
            dot *= 2;
            for (let i = 0; i < m; i++) R[j + i][col] -= dot * x[i];
        }

        // Accumulate Q
        for (let col = 0; col < M; col++) {
            let dot = 0;
            for (let i = 0; i < m; i++) dot += x[i] * Qfull[j + i][col];
            dot *= 2;
            for (let i = 0; i < m; i++) Qfull[j + i][col] -= dot * x[i];
        }
    }

    const Q = transpose2D(Qfull);

    if (mode === 'reduced') {
        const Qr: Float64Array[] = [];
        for (let i = 0; i < M; i++) {
            Qr.push(new Float64Array(K));
            for (let j = 0; j < K; j++) Qr[i][j] = Q[i][j];
        }
        const Rr: Float64Array[] = [];
        for (let i = 0; i < K; i++) {
            Rr.push(new Float64Array(N));
            for (let j = 0; j < N; j++) Rr[i][j] = R[i][j];
        }
        return { Q: Qr, R: Rr };
    }
    return { Q, R };
}

/**
 * QR decomposition.
 * @pytorch torch.linalg.qr
 */
export const qr = async (input: Tensor, mode: 'reduced' | 'complete' = 'reduced'): Promise<{ Q: Tensor; R: Tensor }> => {
    const shape = input.shape;
    const ndim = shape.length;
    if (ndim < 2) throw new Error('linalg.qr: expected tensor with at least 2 dimensions');

    const M = shape[ndim - 2], N = shape[ndim - 1];
    const batch = ndim > 2 ? shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
    const batchShape = ndim > 2 ? shape.slice(0, -2) : [];

    const flatData = await readMatrixToCPU(input);
    const allQ: Float64Array[][] = [], allR: Float64Array[][] = [];

    for (let b = 0; b < batch; b++) {
        const matData = flatData.slice(b * M * N, (b + 1) * M * N);
        const A: Float64Array[] = [];
        for (let i = 0; i < M; i++) A.push(new Float64Array(matData.slice(i * N, (i + 1) * N)));
        const { Q, R } = qr2D(A, mode);
        allQ.push(Q);
        allR.push(R);
    }

    if (batch === 1) {
        return {
            Q: createTensorFromCPU(flattenMatrix(allQ[0]), [allQ[0].length, allQ[0][0].length]),
            R: createTensorFromCPU(flattenMatrix(allR[0]), [allR[0].length, allR[0][0].length]),
        };
    }

    const Qflat = new Float64Array(batch * allQ[0].length * allQ[0][0].length);
    const Rflat = new Float64Array(batch * allR[0].length * allR[0][0].length);
    for (let b = 0; b < batch; b++) {
        Qflat.set(flattenMatrix(allQ[b]), b * allQ[0].length * allQ[0][0].length);
        Rflat.set(flattenMatrix(allR[b]), b * allR[0].length * allR[0][0].length);
    }

    return {
        Q: createTensorFromCPU(Qflat, [...batchShape, allQ[0].length, allQ[0][0].length]),
        R: createTensorFromCPU(Rflat, [...batchShape, allR[0].length, allR[0][0].length]),
    };
};

// ============================================================================
// SOLVE via LU decomposition with partial pivoting
// ============================================================================

function luDecompose(A: Float64Array[]): { L: Float64Array[]; U: Float64Array[]; P: number[] } {
    const n = A.length;
    const U = copy2D(A);
    const L = eye2D(n);
    const P = Array.from({ length: n }, (_, i) => i);

    for (let k = 0; k < n; k++) {
        let maxVal = Math.abs(U[k][k]), maxRow = k;
        for (let i = k + 1; i < n; i++) {
            if (Math.abs(U[i][k]) > maxVal) { maxVal = Math.abs(U[i][k]); maxRow = i; }
        }
        if (maxRow !== k) {
            [U[k], U[maxRow]] = [U[maxRow], U[k]];
            [P[k], P[maxRow]] = [P[maxRow], P[k]];
            for (let j = 0; j < k; j++) [L[k][j], L[maxRow][j]] = [L[maxRow][j], L[k][j]];
        }
        if (Math.abs(U[k][k]) < 1e-15) continue;
        for (let i = k + 1; i < n; i++) {
            L[i][k] = U[i][k] / U[k][k];
            for (let j = k; j < n; j++) U[i][j] -= L[i][k] * U[k][j];
        }
    }
    return { L, U, P };
}

function applyPermMat(B: Float64Array[], P: number[]): Float64Array[] {
    return P.map(i => new Float64Array(B[i]));
}

function forwardSub(L: Float64Array[], B: Float64Array[]): Float64Array[] {
    const n = L.length, m = B[0].length;
    const Y: Float64Array[] = [];
    for (let i = 0; i < n; i++) {
        Y.push(new Float64Array(m));
        for (let j = 0; j < m; j++) {
            let sum = 0;
            for (let k = 0; k < i; k++) sum += L[i][k] * Y[k][j];
            Y[i][j] = B[i][j] - sum;
        }
    }
    return Y;
}

function backSub(U: Float64Array[], Y: Float64Array[]): Float64Array[] {
    const n = U.length, m = Y[0].length;
    const X: Float64Array[] = [];
    for (let i = 0; i < n; i++) X.push(new Float64Array(m));
    for (let i = n - 1; i >= 0; i--) {
        if (Math.abs(U[i][i]) < 1e-15) continue;
        for (let j = 0; j < m; j++) {
            let sum = 0;
            for (let k = i + 1; k < n; k++) sum += U[i][k] * X[k][j];
            X[i][j] = (Y[i][j] - sum) / U[i][i];
        }
    }
    return X;
}

/**
 * Solves the linear system AX = B for X.
 * @pytorch torch.linalg.solve
 */
export const solve = async (input: Tensor, B: Tensor): Promise<Tensor> => {
    const shapeA = input.shape;
    const ndimA = shapeA.length;
    if (ndimA < 2) throw new Error('linalg.solve: expected tensor with at least 2 dimensions');

    const N = shapeA[ndimA - 1], M = shapeA[ndimA - 2];
    if (N !== M) throw new Error('linalg.solve: expected square matrix');

    const shapeB = B.shape;
    const ndimB = shapeB.length;
    const numRHS = ndimB >= 2 ? shapeB[ndimB - 1] : 1;

    const batch = ndimA > 2 ? shapeA.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
    const batchShape = ndimA > 2 ? shapeA.slice(0, -2) : [];

    const flatA = await readMatrixToCPU(input);
    const flatB = await readMatrixToCPU(B);
    const allX: Float64Array[][] = [];

    for (let b = 0; b < batch; b++) {
        const matAData = flatA.slice(b * N * N, (b + 1) * N * N);
        const A: Float64Array[] = [];
        for (let i = 0; i < N; i++) A.push(new Float64Array(matAData.slice(i * N, (i + 1) * N)));

        const matBData = flatB.slice(b * N * numRHS, (b + 1) * N * numRHS);
        const Bmat: Float64Array[] = [];
        for (let i = 0; i < N; i++) Bmat.push(new Float64Array(matBData.slice(i * numRHS, (i + 1) * numRHS)));

        const { L, U, P } = luDecompose(A);
        const PB = applyPermMat(Bmat, P);
        const Y = forwardSub(L, PB);
        const X = backSub(U, Y);
        allX.push(X);
    }

    if (batch === 1) {
        return createTensorFromCPU(flattenMatrix(allX[0]), [N, numRHS]);
    }

    const Xflat = new Float64Array(batch * N * numRHS);
    for (let b = 0; b < batch; b++) Xflat.set(flattenMatrix(allX[b]), b * N * numRHS);
    return createTensorFromCPU(Xflat, [...batchShape, N, numRHS]);
};

// ============================================================================
// PINV - Moore-Penrose pseudo-inverse via SVD
// ============================================================================

/**
 * Moore-Penrose pseudo-inverse.
 * @pytorch torch.linalg.pinv
 */
export const pinv = async (input: Tensor, rcond: number = 1e-15, hermitian: boolean = false): Promise<Tensor> => {
    const shape = input.shape;
    const ndim = shape.length;
    if (ndim < 2) throw new Error('linalg.pinv: expected tensor with at least 2 dimensions');

    const M = shape[ndim - 2], N = shape[ndim - 1];
    const batch = ndim > 2 ? shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
    const batchShape = ndim > 2 ? shape.slice(0, -2) : [];

    const allResult: Float64Array[][] = [];

    if (hermitian) {
        const { eigenvalues, eigenvectors } = await eigh(input);
        const evals = await eigenvalues.toArray();
        const evecsFlat = await eigenvectors.toArray();

        for (let b = 0; b < batch; b++) {
            const V: Float64Array[] = [];
            for (let i = 0; i < N; i++) V.push(new Float64Array(evecsFlat.slice(b * N * N + i * N, b * N * N + (i + 1) * N)));

            const batchEvals = Array.from(evals.slice(b * N, (b + 1) * N)).map(Number);
            const maxAbs = Math.max(...batchEvals.map(Math.abs));
            const threshold = rcond * maxAbs;

            // A^+ = V * diag(invEvals) * V^T
            const VD: Float64Array[] = [];
            for (let i = 0; i < N; i++) {
                VD.push(new Float64Array(N));
                const invEv = Math.abs(batchEvals[i]) > threshold ? 1 / batchEvals[i] : 0;
                for (let j = 0; j < N; j++) VD[i][j] = V[i][j] * invEv;
            }
            const Vt = transpose2D(V);
            allResult.push(matmul2D(VD, Vt));
        }
    } else {
        const { U, S, Vh } = await svd(input, false);
        const sVals = await S.toArray();
        const uFlat = await U.toArray();
        const vhFlat = await Vh.toArray();
        const K = Math.min(M, N);

        for (let b = 0; b < batch; b++) {
            const Umat: Float64Array[] = [];
            for (let i = 0; i < M; i++) Umat.push(new Float64Array(uFlat.slice(b * M * K + i * K, b * M * K + (i + 1) * K)));

            const Vhmat: Float64Array[] = [];
            for (let i = 0; i < K; i++) Vhmat.push(new Float64Array(vhFlat.slice(b * K * N + i * N, b * K * N + (i + 1) * N)));

            const batchS = Array.from(sVals.slice(b * K, (b + 1) * K)).map(Number);
            const threshold = rcond * Math.max(...batchS.map(Math.abs));

            const V = transpose2D(Vhmat);
            const VD: Float64Array[] = [];
            for (let i = 0; i < N; i++) {
                VD.push(new Float64Array(K));
                for (let j = 0; j < K; j++) {
                    VD[i][j] = V[i][j] * (batchS[j] > threshold ? 1 / batchS[j] : 0);
                }
            }
            const Ut = transpose2D(Umat);
            allResult.push(matmul2D(VD, Ut));
        }
    }

    if (batch === 1) {
        return createTensorFromCPU(flattenMatrix(allResult[0]), [N, M]);
    }

    const resultFlat = new Float64Array(batch * N * M);
    for (let b = 0; b < batch; b++) resultFlat.set(flattenMatrix(allResult[b]), b * N * M);
    return createTensorFromCPU(resultFlat, [...batchShape, N, M]);
};

/*
 * ============================================================================
 * USAGE EXAMPLES / TESTS
 * ============================================================================
 *
 * // 1. SVD
 * import * as linalg from './linalg';
 * import { tensor } from './ops/creation';
 *
 * const A = tensor([[3, 2, 2], [2, 3, -2]], { dtype: 'float32' });
 * const { U, S, Vh } = await linalg.svd(A);
 * // Verify: U @ diag(S) @ Vh ≈ A
 *
 *
 * // 2. EIGH (symmetric matrix eigenvalues)
 * const B = tensor([[4, 1], [1, 3]], { dtype: 'float32' });
 * const { eigenvalues, eigenvectors } = await linalg.eigh(B);
 * // eigenvalues sorted ascending; B @ eigenvectors ≈ eigenvectors @ diag(eigenvalues)
 *
 *
 * // 3. QR decomposition
 * const C = tensor([[1, 2], [3, 4], [5, 6]], { dtype: 'float32' });
 * const { Q, R } = await linalg.qr(C, 'reduced');
 * // Verify: Q @ R ≈ C, Q^T @ Q ≈ I
 *
 *
 * // 4. SOLVE linear system AX = B
 * const A_mat = tensor([[2, 1], [1, 3]], { dtype: 'float32' });
 * const B_mat = tensor([[1], [2]], { dtype: 'float32' });
 * const X = await linalg.solve(A_mat, B_mat);
 * // Verify: A @ X ≈ B
 *
 *
 * // 5. PINV (pseudo-inverse)
 * const D = tensor([[1, 2], [3, 4], [5, 6]], { dtype: 'float32' });
 * const D_pinv = await linalg.pinv(D);
 * // Verify: D @ D_pinv @ D ≈ D
 *
 *
 * // Batched:
 * const batched = tensor([[1,2,3,4,5,6,7,8,9,10,11,12]], { shape: [3,2,2], dtype: 'float32' });
 * const { eigenvalues } = await linalg.eigh(batched);
 * // eigenvalues shape: [3, 2]
 */
