import os
import sox

input_dir_name = 'original_data'
output_dir_name = 'splices'
duration_chunks = 10  # in seconds
start_time = 30  # in seconds
downsample_rate = 4000  # in kbps

ds_output_dir_name = 'downsampled_{}'.format(output_dir_name)

if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)
if not os.path.exists(ds_output_dir_name):
    os.makedirs(ds_output_dir_name)

print('Will send spliced audio to {}'.format(output_dir_name))
print('Will send spliced and downsampled audio to' +
      ' {}'.format(ds_output_dir_name))

# Loop over all files within the input directory
for filename in os.listdir(input_dir_name):
    input_filename = os.path.join(input_dir_name, filename)
    if not os.path.isfile(input_filename) or not '.sph' in filename:
        continue
    filename_base = os.path.splitext(filename)[0]

    # This is the total audio track duration less the
    # first and last 30 seconds
    duration = sox.file_info.duration(input_filename) - 60

    n_iterations = int(duration/duration_chunks)

    for i in range(n_iterations):
        # create trasnformer
        splice = sox.Transformer()
        splice_and_downsample = sox.Transformer()

        begin = start_time + i*duration_chunks
        end = begin + duration_chunks
        output_filename = '{}_{}-{}.wav'.format(filename_base, begin, end)
        output_filename = os.path.join(output_dir_name, output_filename)
        ds_output_filename = '{}_{}-{}.wav'.format(filename_base, begin, end)
        ds_output_filename = os.path.join(ds_output_dir_name,
                                          ds_output_filename)

        splice.trim(begin, end)
        splice_and_downsample.trim(begin, end)
        splice_and_downsample.convert(samplerate=downsample_rate)

        splice.build(input_filename, output_filename)
        splice_and_downsample.build(input_filename, ds_output_filename)

        print('Finished split {} of {} for {}'.format(i, n_iterations, 
                                                      filename_base))
