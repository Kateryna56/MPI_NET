using System;
using System.Diagnostics;
using System.Linq;
using MPI;

class Program
{
    static void Main(string[] args)
    {
        using (new MPI.Environment(ref args))
        {
            var comm = Communicator.world;
            int rank = comm.Rank;
            int size = comm.Size;

            int[] sizes = { 100000, 1000000, 10000000, 20000000 };
            double[,] results = new double[sizes.Length, 3];

            for (int i = 0; i < sizes.Length; i++)
            {
                int arraySize = sizes[i];
                int baseSize = arraySize / size;
                int remainder = arraySize % size;

                int[] counts = Enumerable.Repeat(baseSize, size).ToArray();
                for (int j = 0; j < remainder; j++) counts[j]++;
                int[] displs = new int[size];
                for (int j = 1; j < size; j++)
                    displs[j] = displs[j - 1] + counts[j - 1];

                int[] array = null;
                if (rank == 0)
                {
                    array = new int[arraySize];
                    Random rand = new Random(42);
                    for (int j = 0; j < arraySize; j++)
                        array[j] = rand.Next(1_000_000);
                    Console.WriteLine($"Розмiр: масиву {arraySize}");
                }

                int[] localArray = new int[counts[rank]];
                comm.ScatterFromFlattened(array, counts, 0, ref localArray);

                // 1. Max
                Stopwatch sw = null;
                if (rank == 0) sw = Stopwatch.StartNew();
                int localMax = localArray.Max();
                int globalMax = comm.Reduce(localMax, Operation<int>.Max, 0);
                if (rank == 0)
                {
                    sw.Stop();
                    results[i, 0] = sw.Elapsed.TotalSeconds;
                }

                // 2. Sum
                if (rank == 0) sw = Stopwatch.StartNew();
                long localSum = localArray.Select(x => (long)x).Sum();
                long globalSum = comm.Reduce(localSum, Operation<long>.Add, 0);
                if (rank == 0)
                {
                    sw.Stop();
                    results[i, 1] = sw.Elapsed.TotalSeconds;
                }

                // 3. Filter > 900000
                if (rank == 0) sw = Stopwatch.StartNew();
                int[] localFiltered = localArray.Where(x => x > 900000).ToArray();
                int[][] gathered = comm.Gather(localFiltered, 0);
                if (rank == 0)
                {
                    sw.Stop();
                    int totalFiltered = gathered.SelectMany(a => a).Count();
                    results[i, 2] = sw.Elapsed.TotalSeconds;
                }
            }

            if (rank == 0)
            {
                Console.WriteLine("\nРозмiр\tМакс. (с)\tСума (с)\tФiльтрацiя >900000 (с)");
                for (int i = 0; i < sizes.Length; i++)
                {
                    Console.WriteLine($"{sizes[i]}\t{results[i, 0]:0.0000}\t\t{results[i, 1]:0.0000}\t\t{results[i, 2]:0.0000}");
                }
            }

            Console.ReadKey();
        }
    }
}