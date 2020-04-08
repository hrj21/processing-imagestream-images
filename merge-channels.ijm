channels = newArray("Ch1", "Ch6", "Ch9");
colors = newArray("Red", "Blue", "Green");
print ("\\Clear");
dir = getDirectory("input folder");
files = getFileList(dir);
baseChannel = channels[0];
setBatchMode(true);
counter = 0;
for (i=0; i<files.length; i++) {
    merged = false;
    if (endsWith(files[0], "tif")) {
        if (indexOf(files[i], baseChannel) != -1) {
                        print("\\Update0:" + "file " + (i+1) + "/" + files.length + " - MERGING");
    counter++;
    open(dir + "/" + files[i]);
    run(colors[0]);
    names = newArray(channels.length);
    names[0] = files[i];
    for (j=1; j<channels.length; j++) {
         channel = replace(files[i], channels[0], channels[j]);
         open(dir + "/" + channel);
         names[j] = channel;
         run(colors[j]);
    }
    options = "";
    for (j=0; j<names.length;j++) {
        options = options + "c" + (j+1) + "=" + names[j] + " ";
    }
                        options = options + "create";
                        run("Merge Channels...", options);
                        run("Stack to RGB");
                        //run("8-bit");
                        //Stack.setDisplayMode("color");

    result = replace(files[i], channels[0], toString(channels.length) + "_colour");
    saveAs("Tiff", dir + result);
    close();
    merged = true;
                        print(toString(i) + ": file " + files[i] + " - MERGED");
        } 
    } 
    if (!merged) {
        print("\\Update0:" + "file " + (i+1) + "/" + files.length + " - SKIPPING");
        print(toString(i) +": file " + files[i] + " - SKIPPED");
    }
}
print("MERGE CHANNELS FINISHED (" + counter +" hyperstacks created)");
setBatchMode(false);