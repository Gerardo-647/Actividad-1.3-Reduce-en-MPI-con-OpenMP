/* Integrantes (Orden Alfabético):
 * - Cancelada de la O Gerardo Alexander
 * - Hernández Miranda David Alejandro
 */

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <mpi.h>

class OperacionesArreglos {
private:
    int* datos;
    int tamano;

public:
    OperacionesArreglos(int n) : tamano(n) {
        if (n > 0) {
            datos = new int[n];
        }
        else {
            datos = nullptr;
        }
    }

    ~OperacionesArreglos() {
        if (datos != nullptr) {
            delete[] datos;
        }
    }

    int* getDatos() { return datos; }
    int getTamano() { return tamano; }

    void llenarConsecutivo(int offset, const char* equipo, int rank, bool detallado, std::ofstream& logFile) {
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < tamano; i++) {
                int pos_global = offset + i;
                datos[i] = pos_global + 1;
                if (detallado) {
#pragma omp critical
                    {
                        char buffer[512];
                        std::sprintf(buffer, "[Equipo: %s] [Proceso MPI: %d] [Hilo OpenMP: %d] [Posicion: %d] [Operacion: Llenar Secuencial] [Valor: %d]\n",
                            equipo, rank, thread_id, pos_global, datos[i]);
                        std::cout << buffer;
                        logFile << buffer;
                        std::fflush(stdout);
                        logFile.flush();
                    }
                }
            }
        }
    }

    void llenarAleatorio(int rank, unsigned int seed_extra, const char* equipo, bool detallado, std::ofstream& logFile, int offset = 0) {
        std::srand((unsigned int)(std::time(nullptr)) + rank * 997 + seed_extra);
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < tamano; i++) {
                datos[i] = (std::rand() % 1000000) + 1;
                int pos_global = offset + i;
                if (detallado) {
#pragma omp critical
                    {
                        char buffer[512];
                        std::sprintf(buffer, "[Equipo: %s] [Proceso MPI: %d] [Hilo OpenMP: %d] [Posicion: %d] [Operacion: Llenar Aleatorio] [Valor: %d]\n",
                            equipo, rank, thread_id, pos_global, datos[i]);
                        std::cout << buffer;
                        logFile << buffer;
                        std::fflush(stdout);
                        logFile.flush();
                    }
                }
            }
        }
    }

    void sumar(OperacionesArreglos* B, OperacionesArreglos* Res, const char* equipo, int rank, int offset, bool detallado, std::ofstream& logFile) {
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < tamano; i++) {
                Res->datos[i] = this->datos[i] + B->datos[i];
                if (detallado) {
#pragma omp critical
                    {
                        char buffer[512];
                        std::sprintf(buffer, "[Equipo: %s] [Proceso MPI: %d] [Hilo OpenMP: %d] [Posicion: %d] [Operacion: Suma (%d + %d = %d)]\n",
                            equipo, rank, thread_id, offset + i, this->datos[i], B->datos[i], Res->datos[i]);
                        std::cout << buffer;
                        logFile << buffer;
                        std::fflush(stdout);
                        logFile.flush();
                    }
                }
            }
        }
    }

    void restar(OperacionesArreglos* B, OperacionesArreglos* Res, const char* equipo, int rank, int offset, bool detallado, std::ofstream& logFile) {
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < tamano; i++) {
                Res->datos[i] = this->datos[i] - B->datos[i];
                if (detallado) {
#pragma omp critical
                    {
                        char buffer[512];
                        std::sprintf(buffer, "[Equipo: %s] [Proceso MPI: %d] [Hilo OpenMP: %d] [Posicion: %d] [Operacion: Resta (%d - %d = %d)]\n",
                            equipo, rank, thread_id, offset + i, this->datos[i], B->datos[i], Res->datos[i]);
                        std::cout << buffer;
                        logFile << buffer;
                        std::fflush(stdout);
                        logFile.flush();
                    }
                }
            }
        }
    }

    void multiplicar(OperacionesArreglos* B, OperacionesArreglos* Res, const char* equipo, int rank, int offset, bool detallado, std::ofstream& logFile) {
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < tamano; i++) {
                Res->datos[i] = this->datos[i] * B->datos[i];
                if (detallado) {
#pragma omp critical
                    {
                        char buffer[512];
                        std::sprintf(buffer, "[Equipo: %s] [Proceso MPI: %d] [Hilo OpenMP: %d] [Posicion: %d] [Operacion: Multiplicacion (%d * %d = %d)]\n",
                            equipo, rank, thread_id, offset + i, this->datos[i], B->datos[i], Res->datos[i]);
                        std::cout << buffer;
                        logFile << buffer;
                        std::fflush(stdout);
                        logFile.flush();
                    }
                }
            }
        }
    }

    void cuadrado(OperacionesArreglos* Res, const char* equipo, int rank, int offset, bool detallado, std::ofstream& logFile) {
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < tamano; i++) {
                Res->datos[i] = this->datos[i] * this->datos[i];
                if (detallado) {
#pragma omp critical
                    {
                        char buffer[512];
                        std::sprintf(buffer, "[Equipo: %s] [Proceso MPI: %d] [Hilo OpenMP: %d] [Posicion: %d] [Operacion: Cuadrado (%d^2 = %d)]\n",
                            equipo, rank, thread_id, offset + i, this->datos[i], Res->datos[i]);
                        std::cout << buffer;
                        logFile << buffer;
                        std::fflush(stdout);
                        logFile.flush();
                    }
                }
            }
        }
    }

    long long sumaLocalOpenMP() {
        long long suma = 0;
#pragma omp parallel for reduction(+:suma)
        for (int i = 0; i < tamano; i++) {
            suma += datos[i];
        }
        return suma;
    }

    int maxLocalOpenMP() {
        int max_val = (tamano > 0) ? datos[0] : 0;
#pragma omp parallel for reduction(max:max_val)
        for (int i = 0; i < tamano; i++) {
            if (datos[i] > max_val) max_val = datos[i];
        }
        return max_val;
    }

    int minLocalOpenMP() {
        int min_val = (tamano > 0) ? datos[0] : 0;
#pragma omp parallel for reduction(min:min_val)
        for (int i = 0; i < tamano; i++) {
            if (datos[i] < min_val) min_val = datos[i];
        }
        return min_val;
    }

    void mostrar(const char* titulo) {
        std::cout << "\n" << titulo << ": [ ";
        for (int i = 0; i < tamano; i++) {
            std::cout << datos[i] << (i == tamano - 1 ? "" : ", ");
        }
        std::cout << " ]\n";
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size, name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(processor_name, &name_len);

    if (rank == 0) {
        std::cout << "\nIntegrantes:\n 1. Cancelada de la O Gerardo Alexander\n 2. Hernandez Miranda David Alejandro\n\n";
    }

    char log_filename[256];
    std::sprintf(log_filename, "log_equipo_%s_nodo_%d.txt", processor_name, rank);
    std::ofstream logFile(log_filename, std::ios::out);

    int N_total = 40;
    if (argc > 1) {
        N_total = std::atoi(argv[1]);
    }
    bool detallado = (N_total <= 50);
    int N_sub = N_total / size;

    OperacionesArreglos subA(N_sub), subB(N_sub), subRes(N_sub);
    OperacionesArreglos globalA(rank == 0 ? N_total : 0);
    OperacionesArreglos globalB(rank == 0 ? N_total : 0);
    OperacionesArreglos globalRes(rank == 0 ? N_total : 0);

    int opcion = 0;
    do {
        if (rank == 0) {
            std::cout << "\n--- MENU INTERACTIVO (12 Opciones - MPI + OpenMP) ---\n";
            std::cout << "1. Crear / Inicializar Arreglos (Scatter/Gather)\n";
            std::cout << "2. Sumar Arreglos (A + B)\n";
            std::cout << "3. Restar Arreglos (A - B)\n";
            std::cout << "4. Multiplicar Arreglos (A * B)\n";
            std::cout << "5. Calcular Cuadrado de Arreglo (A^2)\n";
            std::cout << "6. Llenar Secuencial\n";
            std::cout << "7. Llenar Aleatorio\n";
            std::cout << "8. Sumatoria Global (Reduce / Allreduce)\n";
            std::cout << "9. Promedio Global\n";
            std::cout << "10. Maximo Global\n";
            std::cout << "11. Minimo Global\n";
            std::cout << "12. Salir\n";
            std::cout << "Seleccione una opcion: ";
            std::cin >> opcion;
            std::cout << "\n";
        }

        MPI_Bcast(&opcion, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (opcion == 12) break;

        double t_inicio = MPI_Wtime();

        if (opcion == 1 || opcion == 6 || opcion == 7) {
            if (rank == 0) {
                if (opcion == 6 || (opcion == 1 && detallado)) {
                    globalA.llenarConsecutivo(0, processor_name, 0, detallado, logFile);
                    globalB.llenarConsecutivo(100, processor_name, 0, detallado, logFile);
                }
                else if (opcion == 7 || (opcion == 1 && !detallado)) {
                    globalA.llenarAleatorio(0, 1, processor_name, detallado, logFile, 0);
                    globalB.llenarAleatorio(0, 2, processor_name, detallado, logFile, 0);
                }
            }
            MPI_Scatter(globalA.getDatos(), N_sub, MPI_INT, subA.getDatos(), N_sub, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Scatter(globalB.getDatos(), N_sub, MPI_INT, subB.getDatos(), N_sub, MPI_INT, 0, MPI_COMM_WORLD);

            if (detallado) {
                if (opcion == 6) {
                    subA.llenarConsecutivo(rank * N_sub, processor_name, rank, detallado, logFile);
                }
                else if (opcion == 7) {
                    subA.llenarAleatorio(rank, (unsigned int)rank + 10, processor_name, detallado, logFile, rank * N_sub);
                }
            }
        }
        else if (opcion >= 2 && opcion <= 5) {
            int offset = rank * N_sub;
            if (opcion == 2) subA.sumar(&subB, &subRes, processor_name, rank, offset, detallado, logFile);
            else if (opcion == 3) subA.restar(&subB, &subRes, processor_name, rank, offset, detallado, logFile);
            else if (opcion == 4) subA.multiplicar(&subB, &subRes, processor_name, rank, offset, detallado, logFile);
            else if (opcion == 5) subA.cuadrado(&subRes, processor_name, rank, offset, detallado, logFile);

            MPI_Gather(subRes.getDatos(), N_sub, MPI_INT, globalRes.getDatos(), N_sub, MPI_INT, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                double t_fin = MPI_Wtime();
                if (detallado) globalRes.mostrar("Arreglo Resultado (Scatter/Gather)");
                std::cout << "\n[INFO] Tiempo de ejecucion operacion: " << (t_fin - t_inicio) << " segundos.\n";
                logFile << "\n[INFO] Tiempo de ejecucion operacion: " << (t_fin - t_inicio) << " segundos.\n";
            }
        }
        else if (opcion >= 8 && opcion <= 11) {
            long long suma_local = subA.sumaLocalOpenMP();
            int max_local = subA.maxLocalOpenMP();
            int min_local = subA.minLocalOpenMP();

            long long suma_global = 0;
            int max_global = 0;
            int min_global = 0;

            MPI_Reduce(&suma_local, &suma_global, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&max_local, &max_global, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&min_local, &min_global, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                double t_fin = MPI_Wtime();
                if (opcion == 8) {
                    std::cout << "[RESULTADO] Sumatoria Total del Arreglo A: " << suma_global << "\n";
                    logFile << "[RESULTADO] Sumatoria Total del Arreglo A: " << suma_global << "\n";
                }
                else if (opcion == 9) {
                    double promedio = (double)suma_global / N_total;
                    std::cout << "[RESULTADO] Promedio Global del Arreglo A: " << promedio << "\n";
                    logFile << "[RESULTADO] Promedio Global del Arreglo A: " << promedio << "\n";
                }
                else if (opcion == 10) {
                    std::cout << "[RESULTADO] Valor Maximo Global del Arreglo A: " << max_global << "\n";
                    logFile << "\n[RESULTADO] Valor Maximo Global del Arreglo A: " << max_global << "\n";
                }
                else if (opcion == 11) {
                    std::cout << "[RESULTADO] Valor Minimo Global del Arreglo A: " << min_global << "\n";
                    logFile << "[RESULTADO] Valor Minimo Global del Arreglo A: " << min_global << "\n";
                }
                std::cout << "[INFO] Tiempo de ejecucion Reduccion: " << (t_fin - t_inicio) << " segundos.\n";
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } while (opcion != 12);

    MPI_Barrier(MPI_COMM_WORLD);

    if (logFile.is_open()) {
        logFile.close();
    }

    if (rank == 0) {
        std::cout << "\nIntegrantes:\n 1. Cancelada de la O Gerardo Alexander\n 2. Hernandez Miranda David Alejandro\n";
    }

    MPI_Finalize();
    return 0;
}