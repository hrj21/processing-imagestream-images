dir1 = getDirectory("Choose Source Directory ");
  format = getFormat();
  dir2 = getDirectory("Choose Destination Directory ");
  list = getFileList(dir1);
  setBatchMode(true);
  for (i=0; i<list.length; i++) {
     showProgress(i+1, list.length);
     open(dir1+list[i]);
     if (format=="8-bit TIFF" || format=="GIF")
        convertTo8Bit();
     saveAs(format, dir2+list[i]);
     close();
  }
 
  function getFormat() {
       formats = newArray("TIFF", "8-bit TIFF", "JPEG", "GIF", "PNG",
          "PGM", "BMP", "FITS", "Text Image", "ZIP", "Raw");
       Dialog.create("Batch Convert");
       Dialog.addChoice("Convert to: ", formats, "TIFF");
       Dialog.show();
       return Dialog.getChoice();
  }

  function convertTo8Bit() {
      if (bitDepth==24)
          run("8-bit Color", "number=256");
      else
          run("8-bit");
  }