# Helper script for creating high-definition videos

The following script can be used to create high-definition videos.
It is not part of the `CoherentStructures.jl` package because we did
not want to introduce additional dependencies.

WARNING: this script will delete/overwrite the file at `output_file`.

Example usage, after the script has been loaded:
```
xs = range(0,stop=10,length=20)
ts = range(0,stop=1.0,length=100)
frames = [
    Plots.plot(xs, x -> sin(t*x),ylim=(-1.0,1.0)) for t in ts
    ]
animatemp4(frames) # Saves to /tmp/output.mp4 by default
```

 <video controls="" height="100%" width="100%">
  <source src="https://raw.githubusercontent.com/natschil/misc/master/videos/sample_video.mp4" type="video/mp4" />
 Your browser does not support the video tag.
 </video>


The script:

```
using Printf,UUIDs

using ProgressMeter,Plots

function animatemp4(fitr,output_file="/tmp/output.mp4",delete_frames_after=true;density=400,framerate=60)
    dirn = @sprintf("/tmp/animate%s/",string(UUIDs.uuid1()))
    mkdir(dirn)
    try
        run(`rm $output_file`)
    catch e
    end
    @showprogress "Saving frames" for (index,i) in enumerate(fitr)
        fname = @sprintf("%s/still%03d.pdf",dirn,index)
        Plots.pdf(i,fname)
        fnamepng = fname[1:(end-3)] * "png"
        run(`convert -density $density $fname $fnamepng`)
    end
    run(`ffmpeg -r 5 -pattern_type glob -i "$dirn/still*.png" -framerate $framerate -pix_fmt yuv420p -c:v libx264 -movflags +faststart -filter:v crop='floor(in_w/2)*2:floor(in_h/2)*2' $output_file`)
    if delete_frames_after
        run(`rm -rf $dirn`)
    else
        println("Individual frames saved at $dirn")
    end
end
```
