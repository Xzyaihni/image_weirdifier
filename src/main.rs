use std::{
    process,
    ops::{Add, AddAssign, Sub, Div, Mul}
};

use argparse::{ArgumentParser, Store};

use image::{RgbImage, Rgb, imageops::FilterType};


fn cry_about_it(text: &str) -> !
{
    println!("error: {text}");

    process::exit(1)
}

#[derive(Debug, Clone, Copy)]
struct Lab
{
    l: f32,
    a: f32,
    b: f32
}

impl Lab
{
    pub fn distance_76(&self, other: Lab) -> f32
    {
        let d_l = other.l - self.l;
        let d_a = other.a - self.a;
        let d_b = other.b - self.b;

        (d_l.powi(2) + d_a.powi(2) + d_b.powi(2)).sqrt()
    }

    /*pub fn distance(&self, other: Lab) -> f32
    {
        let d_l = other.l - self.l;
        let d_a = other.a - self.a;
        let d_b = other.b - self.b;

        let l_term = d_l / ;
        let c_term = ;
        let h_term = ;

        let rotation_term = ;

        let adjust = rotation_term * c_term * h_term;

        (l_term.powi(2) + c_term.powi(2) + h_term.powi(2) + adjust).sqrt()
    }*/
}

impl From<Xyz> for Lab
{
    fn from(value: Xyz) -> Self
    {
        let e = 216.0 / 24389.0;
        let k = 24389.0 / 27.0;

        let convert = |value: f32| -> f32
        {
            if value > e
            {
                value.cbrt()
            } else
            {
                (k * value + 16.0) / 116.0
            }
        };

        let x = convert(value.x / 95.047);
        let y = convert(value.y / 100.0);
        let z = convert(value.z / 108.883);

        let l = 116.0 * y - 16.0;
        let a = 500.0 * (x - y);
        let b = 200.0 * (y - z);

        Self{l, a, b}
    }
}

impl From<Color<u8>> for Lab
{
    fn from(value: Color<u8>) -> Self
    {
        Xyz::from(value).into()
    }
}

#[derive(Debug, Clone, Copy)]
struct Xyz
{
    x: f32,
    y: f32,
    z: f32
}

impl From<Color<u8>> for Xyz
{
    fn from(value: Color<u8>) -> Self
    {
        let linear = |value: u8| -> f32
        {
            let value = value as f32 / u8::MAX as f32;

            let value = if value <= 0.04045
            {
                value / 12.92
            } else
            {
                ((value + 0.055) / 1.055).powf(2.4)
            };

            value * 100.0
        };

        let r = linear(value.r);
        let g = linear(value.g);
        let b = linear(value.b);

        let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
        let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
        let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;

        Self{x, y, z}
    }
}

#[derive(Default, Clone, Copy)]
struct Color<T>
{
    r: T,
    g: T,
    b: T
}

impl<T> Color<T>
{
    pub fn new(r: T, g: T, b: T) -> Self
    {
        Self{r, g, b}
    }
}

impl Color<u8>
{
    pub fn into_rgb8(self) -> Rgb<u8>
    {
        [self.r, self.g, self.b].into()
    }
}

impl From<Color<f32>> for Color<i32>
{
    fn from(value: Color<f32>) -> Self
    {
        let cast = |value: f32| -> i32
        {
            if value < i32::MIN as f32
            {
                i32::MIN
            } else if value > i32::MAX as f32
            {
                i32::MAX
            } else
            {
                value as i32
            }
        };

        Self{r: cast(value.r), g: cast(value.g), b: cast(value.b)}
    }
}

impl From<Color<i32>> for Color<u8>
{
    fn from(value: Color<i32>) -> Self
    {
        let cast = |value: i32| -> u8
        {
            if value < u8::MIN as i32
            {
                u8::MIN
            } else if value > u8::MAX as i32
            {
                u8::MAX
            } else
            {
                value as u8
            }
        };

        Self{r: cast(value.r), g: cast(value.g), b: cast(value.b)}
    }
}

impl From<Rgb<u8>> for Color<u8>
{
    fn from(value: Rgb<u8>) -> Self
    {
        Self{r: value[0], g: value[1], b: value[2]}
    }
}

impl From<Color<i32>> for Color<f32>
{
    fn from(value: Color<i32>) -> Self
    {
        Self{r: value.r as f32, g: value.g as f32, b: value.b as f32}
    }
}

impl From<Color<u8>> for Color<f32>
{
    fn from(value: Color<u8>) -> Self
    {
        Self{r: value.r as f32, g: value.g as f32, b: value.b as f32}
    }
}

impl From<Color<u8>> for Color<i32>
{
    fn from(value: Color<u8>) -> Self
    {
        Self{r: value.r as i32, g: value.g as i32, b: value.b as i32}
    }
}

impl<T: Add<Output=T>> Add for Color<T>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output
    {
        Self::new(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b)
    }
}

impl<T: AddAssign> AddAssign for Color<T>
{
    fn add_assign(&mut self, rhs: Self)
    {
        self.r += rhs.r;
        self.g += rhs.g;
        self.b += rhs.b;
    }
}

impl<T: Sub<Output=T>> Sub for Color<T>
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output
    {
        Self::new(self.r - rhs.r, self.g - rhs.g, self.b - rhs.b)
    }
}

impl<T: Div<Output=T> + Copy> Div<T> for Color<T>
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output
    {
        Self::new(self.r / rhs, self.g / rhs, self.b / rhs)
    }
}

impl<T: Mul<Output=T> + Copy> Mul<T> for Color<T>
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output
    {
        Self::new(self.r * rhs, self.g * rhs, self.b * rhs)
    }
}

fn take_closest_color(
    pallete_raw: &mut Vec<Color<u8>>,
    pallete: &mut Vec<Lab>,
    pixel: Color<u8>
) -> Color<u8>
{
    let mut lowest_index = 0;

    let pixel: Lab = pixel.into();
    let mut lowest_distance = pallete[lowest_index].distance_76(pixel);

    for (index, color) in pallete.iter().enumerate().skip(1)
    {
        let distance = color.distance_76(pixel);

        if distance < lowest_distance
        {
            lowest_index = index;
            lowest_distance = distance;

            if distance == 0.0
            {
                break;
            }
        }
    }

    pallete.remove(lowest_index);
    pallete_raw.remove(lowest_index)
}

fn palletify(image: &mut RgbImage, mut pallete_raw: Vec<Color<u8>>, mut pallete: Vec<Lab>)
{
    let total_size = image.width() * image.height();

    let mut index_list = (0..total_size).collect::<Vec<_>>();
    fastrand::shuffle(&mut index_list);

    let mut errors = (0..total_size).map(|_| Default::default()).collect::<Vec<Color<i32>>>();

    let (width, height) = (image.width() as i32, image.height() as i32);
    let calculate_index = |x: i32, y: i32| -> Option<usize>
    {
        if !(0..width).contains(&x) || !(0..height).contains(&y)
        {
            None
        } else
        {
            Some((y * width + x) as usize)
        }
    };

    index_list.into_iter().for_each(|index|
    {
        let x = index % image.width();
        let y = index / image.width();

        let pixel = image.get_pixel_mut(x, y);

        let color: Color<u8> =
        {
            let pixel_index = calculate_index(x as i32, y as i32).unwrap();

            let pixel = Color::<u8>::from(*pixel);

            Color::<i32>::from(pixel) + errors[pixel_index]
        }.into();

        let closest_color = take_closest_color(&mut pallete_raw, &mut pallete, color);

        let neighbors_div = 12;
        let error: Color<f32> = (color - closest_color).into();
        let error = error / neighbors_div as f32;

        let neighbors = [
            ((1, 0), 2),
            ((-1, 0), 2),
            ((0, 1), 2),
            ((0, -1), 2),
            ((1, -1), 1),
            ((-1, -1), 1),
            ((1, 1), 1),
            ((-1, 1), 1)
        ];

        for neighbor in neighbors
        {
            let (x, y) = neighbor.0;

            if let Some(neighbor_index) = calculate_index(x, y)
            {
                let error = error * neighbor.1 as f32;

                errors[neighbor_index] += error.into();
            }
        }

        *pixel = closest_color.into_rgb8();
    });
}

fn main()
{
    let mut input_path = String::new();

    {
        let mut parser = ArgumentParser::new();

        parser.refer(&mut input_path)
            .add_option(&["-i", "--input"], Store, "input image path")
            .add_argument("input_path", Store, "input image path")
            .required();

        parser.parse_args_or_exit();
    }

    let image = image::open(input_path).unwrap_or_else(|err|
    {
        cry_about_it(&format!("image error {err:?}"))
    });

    let (new_width, new_height) = if image.width() < image.height()
    {
        (128, 256)
    } else
    {
        (256, 128)
    };

    let mut image = image.resize_exact(new_width, new_height, FilterType::Lanczos3).into_rgb8();

    const MAX_COLOR: usize = 256 / 8;
    const COLORS_AMOUNT: usize = MAX_COLOR.pow(3);

    let (pallete_raw, pallete) = (0..COLORS_AMOUNT).map(|index|
    {
        let r = (index / MAX_COLOR.pow(2)) * 8;
        let g = ((index / MAX_COLOR) % MAX_COLOR) * 8;
        let b = (index % MAX_COLOR) * 8;

        let color = Color::new(r as u8, g as u8, b as u8);
        (color, Lab::from(color))
    }).unzip();

    /*image.pixels_mut().enumerate().for_each(|(index, pixel)|
    {
        *pixel = pallete[index];
    });*/
    palletify(&mut image, pallete_raw, pallete);

    image.save("output.png").unwrap_or_else(|err|
    {
        cry_about_it(&format!("problem saving the image {err:?}"))
    });
}

#[cfg(test)]
mod tests
{
    use super::*;

    fn close_enough(value0: f32, value1: f32) -> bool
    {
        let epsilon = 0.01;

        eprintln!("comparing {value0:.3} and {value1:.3}");
        (value1 - value0).abs() < epsilon
    }

    #[test]
    fn xyz()
    {
        let color = Color::new(17, 171, 109);
        let xyz: Xyz = color.into();

        assert!(close_enough(xyz.x, 17.55));
        assert!(close_enough(xyz.y, 30.34));
        assert!(close_enough(xyz.z, 19.40));
    }

    #[test]
    fn lab()
    {
        let color = Color::new(17, 171, 109);
        let lab: Lab = color.into();

        assert!(close_enough(lab.l, 61.95));
        assert!(close_enough(lab.a, -51.27));
        assert!(close_enough(lab.b, 21.86));
    }
}