import os
import sox

input_dir_name_base = '/home/paperspace/Documents' +\
                      '/TEDLIUM/TEDLIUM_release2/{}/sph'
input_dir_name_dirs = ['dev', 'test', 'train']
duration_chunks = 2  # in seconds
start_time = 30  # in seconds
end_time = -30  # in seconds
downsample_rate = 4000  # in kbps
output_dir_name_base = '/home/paperspace/Documents' +\
                       '/TEDLIUM/TEDLIUM_release2/preprocessed'

output_dir_name = os.path.join(output_dir_name_base, 'splices')
ds_output_dir_name = os.path.join(output_dir_name_base, 'downsampled_splices')

if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)
if not os.path.exists(ds_output_dir_name):
    os.makedirs(ds_output_dir_name)

print('Will send spliced audio to {}'.format(output_dir_name))
print('Will send spliced and downsampled audio to' +
      ' {}'.format(ds_output_dir_name))

for input_dir_name_dir in input_dir_name_dirs:
    input_dir_name = input_dir_name_base.format(input_dir_name_dir)

    # Loop over all files within the input directory
    for filename in os.listdir(input_dir_name):
        input_filename = os.path.join(input_dir_name, filename)
        if not os.path.isfile(input_filename) or '.sph' not in filename:
            continue
        filename_base = os.path.splitext(filename)[0]

        # This is the total audio track duration less the
        # start and end times
        duration = sox.file_info.duration(input_filename) - (start_time -
                                                             end_time)

        n_iterations = int(duration/duration_chunks)

        for i in range(n_iterations):
            # create trasnformer
            splice = sox.Transformer()
            splice_and_downsample = sox.Transformer()

            begin = start_time + i*duration_chunks
            end = begin + duration_chunks
            output_filename = '{}_{}-{}.wav'.format(filename_base, begin, end)
            output_filename = os.path.join(output_dir_name, output_filename)
            ds_output_filename = '{}_{}-{}.wav'.format(filename_base,
                                                       begin, end)
            ds_output_filename = os.path.join(ds_output_dir_name,
                                              ds_output_filename)

            splice.trim(begin, end)
            splice_and_downsample.trim(begin, end)
            splice_and_downsample.convert(samplerate=downsample_rate)

            splice.build(input_filename, output_filename)
            splice_and_downsample.build(input_filename, ds_output_filename)

            print('Finished split {} of {} for {}'.format(i + 1, n_iterations,
                                                          filename_base))
