import os
import json
import tqdm
import sox

splice_settings_file = '../settings/data_settings.json'

settings = json.load(open(splice_settings_file))
input_dir_name_base = settings['input_dir_name_base']
input_dir_name_dirs = settings['input_dir_name_dirs']
splice_duration = settings['splice_duration']
start_time = settings['start_time']
end_time = settings['end_time']
downsample_rate = settings['downsample_rate']
output_dir_name_base = settings['output_dir_name_base']

output_dir_name = os.path.join(output_dir_name_base, 'splices')
ds_output_dir_name = os.path.join(output_dir_name_base, 'downsampled_splices')
output_data_info_file_name = os.path.join(output_dir_name_base,
                                          'data_info.json')

if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)
if not os.path.exists(ds_output_dir_name):
    os.makedirs(ds_output_dir_name)

print('Will send spliced audio to {}'.format(output_dir_name))
print('Will send spliced and downsampled audio to' +
      ' {}'.format(ds_output_dir_name))
print('Will write data info to {}'.format(output_data_info_file_name))

processed_data_info = settings
processed_data_info['original_bitrate'] = None

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
        if processed_data_info['original_bitrate'] is None:
            processed_data_info['original_bitrate'] =\
                sox.file_info.bitrate(input_filename)
            if 'kb' in processed_data_info['sampling_rate_units']:
                processed_data_info['original_bitrate'] *= 1000

        n_iterations = int(duration/splice_duration)
        num_of_digits = len(str(int(duration)))
        num_format = '{{:0{}d}}'.format(num_of_digits)
        file_name_template = '{{}}_{}-{}.wav'.format(num_format, num_format)

        print('On file {}'.format(filename_base))
        for i in tqdm.trange(n_iterations):
            # create trasnformer
            splice = sox.Transformer()
            splice_and_downsample = sox.Transformer()

            begin = int(start_time + i*splice_duration)
            end = int(begin + splice_duration)
            output_filename = file_name_template.format(filename_base,
                                                        begin, end)
            output_filename = os.path.join(output_dir_name, output_filename)
            ds_output_filename = file_name_template.format(filename_base,
                                                           begin, end)
            ds_output_filename = os.path.join(ds_output_dir_name,
                                              ds_output_filename)

            splice.trim(begin, end)
            splice_and_downsample.trim(begin, end)
            splice_and_downsample.convert(samplerate=downsample_rate)

            splice.build(input_filename, output_filename)
            splice_and_downsample.build(input_filename, ds_output_filename)

with open(output_data_info_file_name, 'w') as outfile:
    json.dump(processed_data_info, outfile)
