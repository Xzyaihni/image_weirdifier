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

impl Color<f32>
{
    pub fn distance(&self, value: Color<f32>) -> f32
    {
        let diff = *self - value;

        ((diff.r).powi(2) + (diff.g).powi(2) + (diff.b).powi(2)).sqrt()
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

fn take_closest_color(pallete: &mut Vec<Color<u8>>, pixel: Color<u8>) -> Color<u8>
{
    let mut lowest_index = 0;

    let mut lowest_distance = Color::<f32>::from(pallete[lowest_index]).distance(pixel.into());

    for (index, color) in pallete.iter().enumerate().skip(1)
    {
        let distance = Color::<f32>::from(*color).distance(pixel.into());

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

    pallete.remove(lowest_index)
}

fn palletify(image: &mut RgbImage, mut pallete: Vec<Color<u8>>)
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

        let closest_color = take_closest_color(&mut pallete, color);

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

    let pallete = (0..COLORS_AMOUNT).map(|index|
    {
        let r = (index / MAX_COLOR.pow(2)) * 8;
        let g = ((index / MAX_COLOR) % MAX_COLOR) * 8;
        let b = (index % MAX_COLOR) * 8;

        Color::new(r as u8, g as u8, b as u8)
    }).collect::<Vec<Color<u8>>>();

    /*image.pixels_mut().enumerate().for_each(|(index, pixel)|
    {
        *pixel = pallete[index];
    });*/
    palletify(&mut image, pallete);

    image.save("output.png").unwrap_or_else(|err|
    {
        cry_about_it(&format!("problem saving the image {err:?}"))
    });
}
