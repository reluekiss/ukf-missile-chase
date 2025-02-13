#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TIMESTEPS 1000
#define STATE_DIM 9
#define MEAS_DIM 3
#define N_SIGMA (2*STATE_DIM+1)

// Box-Muller normal random generator
float rand_normal(void) {
    static int haveSpare = 0;
    static float spare;
    if(haveSpare) {
        haveSpare = 0;
        return spare;
    }
    haveSpare = 1;
    float u, v, s;
    do {
        u = (rand()/(float)RAND_MAX)*2.0f - 1.0f;
        v = (rand()/(float)RAND_MAX)*2.0f - 1.0f;
        s = u*u + v*v;
    } while(s >= 1.0f || s == 0.0f);
    s = sqrtf(-2.0f*logf(s)/s);
    spare = v * s;
    return u * s;
}

// Invert 3x3 matrix
int invert3x3(float m[3][3], float invOut[3][3]) {
    float det = m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
              - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
              + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
    if(fabs(det) < 1e-6) return 0;
    float invDet = 1.0f/det;
    invOut[0][0] =  (m[1][1]*m[2][2]-m[1][2]*m[2][1])*invDet;
    invOut[0][1] = -(m[0][1]*m[2][2]-m[0][2]*m[2][1])*invDet;
    invOut[0][2] =  (m[0][1]*m[1][2]-m[0][2]*m[1][1])*invDet;
    invOut[1][0] = -(m[1][0]*m[2][2]-m[1][2]*m[2][0])*invDet;
    invOut[1][1] =  (m[0][0]*m[2][2]-m[0][2]*m[2][0])*invDet;
    invOut[1][2] = -(m[0][0]*m[1][2]-m[0][2]*m[1][0])*invDet;
    invOut[2][0] =  (m[1][0]*m[2][1]-m[1][1]*m[2][0])*invDet;
    invOut[2][1] = -(m[0][0]*m[2][1]-m[0][1]*m[2][0])*invDet;
    invOut[2][2] =  (m[0][0]*m[1][1]-m[0][1]*m[1][0])*invDet;
    return 1;
}

// Cholesky decomposition for a 9x9 matrix: A = L*L^T; returns 0 if not PD.
int cholesky_decomposition(float A[STATE_DIM][STATE_DIM], float L[STATE_DIM][STATE_DIM]) {
    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = A[i][j];
            for (int k = 0; k < j; k++) {
                sum -= L[i][k]*L[j][k];
            }
            if(i == j) {
                if(sum <= 0) return 0;
                L[i][j] = sqrtf(sum);
            } else {
                L[i][j] = sum / L[j][j];
            }
        }
        for (int j = i+1; j < STATE_DIM; j++)
            L[i][j] = 0.0f;
    }
    return 1;
}

int main(void)
{
    const int screenWidth = 800, screenHeight = 600;
    InitWindow(screenWidth, screenHeight, "Missile Guidance with UKF (Accel Model)");

    Camera3D camera = { 0 };
    camera.position = (Vector3){ 0.0f, -150.0f, 100.0f };
    camera.target   = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up       = (Vector3){ 0.0f, 0.0f, 1.0f };
    camera.fovy     = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    SetTargetFPS(60);

    float dt = 0.1f, omega_target = 0.1f;
    int timesteps = TIMESTEPS;

    float x_target = 50.0f, y_target = 50.0f, z_target = 0.0f;
    float vx_target = 5.0f, vy_target = 2.0f, vz_target = 0.0f;

    float x_missile = 0.0f, y_missile = 0.0f, z_missile = 0.0f;

    float Q[STATE_DIM][STATE_DIM] = {0};
    for (int i = 0; i < STATE_DIM; i++) Q[i][i] = 1.0f;

    float R_IR[MEAS_DIM][MEAS_DIM] = {0};
    for (int i = 0; i < MEAS_DIM; i++) R_IR[i][i] = 1.0f;

    // (constant acceleration model)
    float A[STATE_DIM][STATE_DIM] = {0};
    for (int i = 0; i < STATE_DIM; i++) A[i][i] = 1.0f;
    A[0][3] = dt;              A[1][4] = dt;              A[2][5] = dt;
    A[0][6] = 0.5f * dt * dt;  A[1][7] = 0.5f * dt * dt;  A[2][8] = 0.5f * dt * dt;
    A[3][6] = dt;              A[4][7] = dt;              A[5][8] = dt;

    // (target state relative to missile).
    float x_hat[STATE_DIM] = { x_target, y_target, z_target,
                               vx_target, vy_target, vz_target,
                               0.0f, 0.0f, 0.0f };

    float P[STATE_DIM][STATE_DIM] = {0};
    for (int i = 0; i < STATE_DIM; i++) P[i][i] = 1.0f;

    float estimated_state[TIMESTEPS][STATE_DIM] = {0};
    float target_state[TIMESTEPS][MEAS_DIM] = {0};
    float measurements[TIMESTEPS][MEAS_DIM] = {0};
    float innovation_norms[TIMESTEPS] = {0};

    srand(time(NULL));

    // UKF stuff 
    float alpha = 0.005f, beta = 2.0f, kappa = 0.0f;
    float lambda = (alpha*alpha)*(STATE_DIM + kappa) - STATE_DIM;
    float gamma = sqrtf(STATE_DIM + lambda);
    float wm[N_SIGMA], wc[N_SIGMA];
    wm[0] = lambda/(STATE_DIM+lambda);
    wc[0] = wm[0] + (1 - alpha*alpha + beta);
    for (int i = 1; i < N_SIGMA; i++) {
        wm[i] = 1.0f/(2*(STATE_DIM+lambda));
        wc[i] = wm[i];
    }

    for (int k = 0; k < timesteps; k++) {
        x_target += vx_target * dt + 5.0f * sinf(omega_target * k * dt);
        y_target += vy_target * dt + 5.0f * cosf(omega_target * k * dt);
        z_target += vz_target * dt;
        target_state[k][0] = x_target;
        target_state[k][1] = y_target;
        target_state[k][2] = z_target;

        float meas[MEAS_DIM] = {
            (x_target - x_missile) + rand_normal(),
            (y_target - y_missile) + rand_normal(),
            (z_target - z_missile) + rand_normal()
        };
        for (int i = 0; i < MEAS_DIM; i++) measurements[k][i] = meas[i];

        float sigma_pts[N_SIGMA][STATE_DIM];
        float L[STATE_DIM][STATE_DIM] = {0};
        if(!cholesky_decomposition(P, L))
            continue;
        for (int i = 0; i < STATE_DIM; i++)
            sigma_pts[0][i] = x_hat[i];
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                float offset = gamma * L[j][i];
                sigma_pts[i+1][j] = x_hat[j] + offset;
                sigma_pts[i+1+STATE_DIM][j] = x_hat[j] - offset;
            }
        }
        // propagate sigma points through process model: x = A*x
        float sigma_pred[N_SIGMA][STATE_DIM] = {0};
        for (int i = 0; i < N_SIGMA; i++) {
            for (int r = 0; r < STATE_DIM; r++) {
                for (int c = 0; c < STATE_DIM; c++) {
                    sigma_pred[i][r] += A[r][c] * sigma_pts[i][c];
                }
            }
        }
        float x_pred[STATE_DIM] = {0};
        for (int i = 0; i < N_SIGMA; i++) {
            for (int j = 0; j < STATE_DIM; j++)
                x_pred[j] += wm[i] * sigma_pred[i][j];
        }
        float P_pred[STATE_DIM][STATE_DIM] = {0};
        for (int i = 0; i < N_SIGMA; i++) {
            float diff[STATE_DIM];
            for (int j = 0; j < STATE_DIM; j++)
                diff[j] = sigma_pred[i][j] - x_pred[j];
            for (int r = 0; r < STATE_DIM; r++)
                for (int c = 0; c < STATE_DIM; c++)
                    P_pred[r][c] += wc[i] * diff[r] * diff[c];
        }
        // add process noise
        for (int i = 0; i < STATE_DIM; i++)
            for (int j = 0; j < STATE_DIM; j++)
                P_pred[i][j] += Q[i][j];

        // transform sigma points into measurement space: h(x) = first 3 elements
        float sigma_meas[N_SIGMA][MEAS_DIM] = {0};
        for (int i = 0; i < N_SIGMA; i++) {
            for (int j = 0; j < MEAS_DIM; j++)
                sigma_meas[i][j] = sigma_pred[i][j];
        }
        // predicted measurement mean
        float y_pred[MEAS_DIM] = {0};
        for (int i = 0; i < N_SIGMA; i++) {
            for (int j = 0; j < MEAS_DIM; j++)
                y_pred[j] += wm[i] * sigma_meas[i][j];
        }
        // measurement covariance S
        float S[MEAS_DIM][MEAS_DIM] = {0};
        for (int i = 0; i < N_SIGMA; i++) {
            float diff[MEAS_DIM];
            for (int j = 0; j < MEAS_DIM; j++)
                diff[j] = sigma_meas[i][j] - y_pred[j];
            for (int r = 0; r < MEAS_DIM; r++)
                for (int c = 0; c < MEAS_DIM; c++)
                    S[r][c] += wc[i] * diff[r] * diff[c];
        }
        // add measurement noise
        for (int i = 0; i < MEAS_DIM; i++)
            for (int j = 0; j < MEAS_DIM; j++)
                S[i][j] += R_IR[i][j];
        // cross covariance Px_y (STATE_DIM x MEAS_DIM)
        float Px_y[STATE_DIM][MEAS_DIM] = {0};
        for (int i = 0; i < N_SIGMA; i++) {
            float diff_x[STATE_DIM], diff_y[MEAS_DIM];
            for (int j = 0; j < STATE_DIM; j++)
                diff_x[j] = sigma_pred[i][j] - x_pred[j];
            for (int j = 0; j < MEAS_DIM; j++)
                diff_y[j] = sigma_meas[i][j] - y_pred[j];
            for (int r = 0; r < STATE_DIM; r++)
                for (int c = 0; c < MEAS_DIM; c++)
                    Px_y[r][c] += wc[i] * diff_x[r] * diff_y[c];
        }
        // kalman gain K = Px_y * S_inv
        float S_inv[MEAS_DIM][MEAS_DIM];
        if(!invert3x3(S, S_inv))
            continue;
        float K_gain[STATE_DIM][MEAS_DIM] = {0};
        for (int i = 0; i < STATE_DIM; i++)
            for (int j = 0; j < MEAS_DIM; j++)
                for (int k2 = 0; k2 < MEAS_DIM; k2++)
                    K_gain[i][j] += Px_y[i][k2] * S_inv[k2][j];
        // (measurement residual)
        float innov[MEAS_DIM];
        for (int i = 0; i < MEAS_DIM; i++)
            innov[i] = meas[i] - y_pred[i];
        float innovation_norm = 0.0f;
        for (int i = 0; i < MEAS_DIM; i++)
            innovation_norm += innov[i]*innov[i];
        innovation_norm = sqrtf(innovation_norm);
        innovation_norms[k] = innovation_norm;
        // update state estimate: x_hat = x_pred + K*(innovation)
        for (int i = 0; i < STATE_DIM; i++) {
            float sum = 0.0f;
            for (int j = 0; j < MEAS_DIM; j++)
                sum += K_gain[i][j] * innov[j];
            x_hat[i] = x_pred[i] + sum;
        }
        // update covariance: P = P_pred - K * S * K^T
        float KS[STATE_DIM][MEAS_DIM] = {0};
        for (int i = 0; i < STATE_DIM; i++)
            for (int j = 0; j < MEAS_DIM; j++)
                for (int k2 = 0; k2 < MEAS_DIM; k2++)
                    KS[i][j] += K_gain[i][k2] * S[k2][j];
        float tempP[STATE_DIM][STATE_DIM] = {0};
        for (int i = 0; i < STATE_DIM; i++)
            for (int j = 0; j < STATE_DIM; j++)
                tempP[i][j] = P_pred[i][j];
        for (int i = 0; i < STATE_DIM; i++)
            for (int j = 0; j < STATE_DIM; j++)
                for (int k2 = 0; k2 < MEAS_DIM; k2++)
                    tempP[i][j] -= KS[i][k2] * K_gain[j][k2];
        for (int i = 0; i < STATE_DIM; i++)
            for (int j = 0; j < STATE_DIM; j++)
                P[i][j] = tempP[i][j];

        for (int i = 0; i < STATE_DIM; i++)
            estimated_state[k][i] = x_hat[i];
    }

    int simIndex = 0;
    float timer = 0.0f;
    while (!WindowShouldClose())
    {
        float deltaTime = GetFrameTime();
        timer += deltaTime;
        if(timer >= dt && simIndex < timesteps - 1) { timer = 0.0f; simIndex++; }
        
        BeginDrawing();
            ClearBackground(RAYWHITE);
            BeginMode3D(camera);
                DrawGrid(20, 5.0f);
                for (int i = 0; i < simIndex - 1; i++) {
                    Vector3 p1 = { estimated_state[i][0], estimated_state[i][1], estimated_state[i][2] };
                    Vector3 p2 = { estimated_state[i+1][0], estimated_state[i+1][1], estimated_state[i+1][2] };
                    DrawLine3D(p1, p2, RED);
                }
                Vector3 targetPos = { target_state[simIndex][0], target_state[simIndex][1], target_state[simIndex][2] };
                DrawSphere(targetPos, 2.0f, BLUE);
                Vector3 estPos = { estimated_state[simIndex][0], estimated_state[simIndex][1], estimated_state[simIndex][2] };
                camera.target = estPos;
                camera.position = (Vector3){ estPos.x - 250, estPos.y - 250, estPos.z + 130 };
                DrawSphere(estPos, 1.5f, RED);
            EndMode3D();
            DrawText("Missile Guidance with UKF (Acceleration Model)", 10, 10, 20, DARKGRAY);
            char stateText[256];
            sprintf(stateText, "State: [%.2f, %.2f, %.2f, ...]", estimated_state[simIndex][0], estimated_state[simIndex][1], estimated_state[simIndex][2]);
            DrawText(stateText, 10, 40, 20, DARKGRAY);
            char measText[128];
            sprintf(measText, "Measurement: [%.2f, %.2f, %.2f]", measurements[simIndex][0], measurements[simIndex][1], measurements[simIndex][2]);
            DrawText(measText, 10, 70, 20, DARKGRAY);
            char innovText[128];
            sprintf(innovText, "Innovation Norm: %.2f", innovation_norms[simIndex]);
            DrawText(innovText, 10, 100, 20, DARKGRAY);
        EndDrawing();
    }
    CloseWindow();
    return 0;
}
