#ifndef IMAGEREQUESTER_H
#define IMAGEREQUESTER_H

#include <Chip.h>
#include <QImage>
#include <QPainter>
#include <QMap>

#include "PixelData.h"

class ImageRequester
{
public:
    ImageRequester(Chip* c = 0);

    PixelData layerPixelData(QString layerName);
    QImage layer(QString layerName);

private:
    Chip* chip;
};

#endif // IMAGEREQUESTER_H
