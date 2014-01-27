#include "ImageRequester.h"

#include "PainterLG.h"

ImageRequester::ImageRequester(Chip* c)
{
    chip = c;
}

PixelData ImageRequester::layerPixelData(QString layerName)
{

}

QImage ImageRequester::layer(QString layerName)
{
    QImage ret(chip->boundingRect(layerName).size(), QImage::Format_Mono);
    QPainter p;
    p.begin(&ret);
    chip->render(new PainterLG(&p), layerName);
    p.end();
    return ret;
}
