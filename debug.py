import ray

ds = ray.data.read_binary_files("gs://assemble-sam2/data/sav_000.tar").repartition(10).limit(1)

print(ds.take_all())