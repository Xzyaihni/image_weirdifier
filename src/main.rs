use std::{
    process,
    ops::{Add, AddAssign, Sub, Div, Mul}
};

use argparse::{ArgumentParser, Store, StoreOption, StoreTrue};

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
    #[allow(dead_code)]
    pub fn distance_76(&self, other: Lab) -> f32
    {
        let d_l = other.l - self.l;
        let d_a = other.a - self.a;
        let d_b = other.b - self.b;

        d_l.powi(2) + d_a.powi(2) + d_b.powi(2)
    }

    pub fn distance(&self, other: Lab) -> f32
    {
        let d_l = other.l - self.l;

        let c0 = (self.a.powi(2) + self.b.powi(2)).sqrt();
        let c1 = (other.a.powi(2) + other.b.powi(2)).sqrt();

        let k_l = 1.0;
        let k_c = 1.0;
        let k_h = 1.0;

        let l_mean = (self.l + other.l) / 2.0;
        let c_mean = (c0 + c1) / 2.0;

        let g = 1.0 + 0.5 * (1.0 - (c_mean.powi(7) / (c_mean.powi(7) + 25.0_f32.powi(7))).sqrt());

        let a0_strike = self.a * g;
        let a1_strike = other.a * g;

        let c0_strike = (a0_strike.powi(2) + self.b.powi(2)).sqrt();
        let c1_strike = (a1_strike.powi(2) + other.b.powi(2)).sqrt();

        let d_c = c1_strike - c0_strike;

        let c_strike_mean = (c0_strike + c1_strike) / 2.0;

        let checked_atan = |y: f32, x: f32|
        {
            if y == 0.0 && x == 0.0
            {
                0.0
            } else
            {
                y.atan2(x)
            }
        };

        let positify = |radians: f32|
        {
            let degrees = radians.to_degrees();
            if degrees < 0.0
            {
                degrees + 360.0
            } else
            {
                degrees
            }
        };

        let h0 = positify(checked_atan(self.b, a0_strike));
        let h1 = positify(checked_atan(other.b, a1_strike));

        let l_mean_shifted = (l_mean - 50.0).powi(2);

        let temp_h_diff = (h0 - h1).abs();

        let d_small_h = if c0_strike == 0.0 || c1_strike == 0.0
        {
            0.0
        } else
        {
            if temp_h_diff <= 180.0
            {
                h1 - h0
            } else if h1 <= h0
            {
                h1 - h0 + 360.0
            } else
            {
                h1 - h0 - 360.0
            }
        };

        let d_h = 2.0 * (c0_strike * c1_strike).sqrt() * (d_small_h.to_radians() / 2.0).sin();

        let h_mean = if c0_strike == 0.0 || c1_strike == 0.0
        {
            h0 + h1
        } else
        {
            if temp_h_diff <= 180.0
            {
                (h0 + h1) / 2.0
            } else if (h0 + h1) < 360.0
            {
                (h0 + h1 + 360.0) / 2.0
            } else
            {
                (h0 + h1 - 360.0) / 2.0
            }
        }.to_radians();

        let t = 1.0
            - 0.17 * (h_mean - 30.0_f32.to_radians()).cos()
            + 0.24 * (2.0 * h_mean).cos()
            + 0.32 * (3.0 * h_mean + 6.0_f32.to_radians()).cos()
            - 0.20 * (4.0 * h_mean - 63.0_f32.to_radians()).cos();

        let l_scale = 1.0 + (0.015 * l_mean_shifted) / ((20.0 + l_mean_shifted).sqrt());
        let c_scale = 1.0 + 0.045 * c_strike_mean;
        let h_scale = 1.0 + 0.015 * c_strike_mean * t;

        let l_term = d_l / (l_scale * k_l);
        let c_term = d_c / (c_scale * k_c);
        let h_term = d_h / (h_scale * k_h);

        let r_adjust = (c_strike_mean.powi(7) / (c_strike_mean.powi(7) + 25.0_f32.powi(7))).sqrt();
        let r_theta = (h_mean - 275.0_f32.to_radians()) / 25.0_f32.to_radians();

        let rotation_term = -2.0 * r_adjust
            * (60.0_f32.to_radians() * (-r_theta.powi(2)).exp()).sin();

        /*dbg!(
            self.l, self.a, self.b,
            other.l, other.a, other.b,
            temp_h_diff,
            h0, h1,
            l_scale, c_scale, h_scale,
            h_mean, d_small_h,
            t, rotation_term,
            g - 1.0,
            a0_strike, a1_strike,
            c0_strike, c1_strike,
            l_term, c_term, h_term,
            d_l, d_c, d_h
        );*/

        let adjust = rotation_term * c_term * h_term;

        (l_term.powi(2) + c_term.powi(2) + h_term.powi(2) + adjust).sqrt()
    }
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
    pixel: Color<u8>,
    fast_dist: bool
) -> Color<u8>
{
    let mut lowest_index = 0;

    let pixel: Lab = pixel.into();

    let calc_distance = |color: Lab, other: Lab| -> f32
    {
        if fast_dist
        {
            color.distance_76(other)
        } else
        {
            color.distance(other)
        }
    };

    let mut lowest_distance = calc_distance(pallete[lowest_index], pixel);

    for (index, color) in pallete.iter().enumerate().skip(1)
    {
        let distance = calc_distance(*color, pixel);

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

fn palletify(
    image: &mut RgbImage,
    mut pallete_raw: Vec<Color<u8>>,
    mut pallete: Vec<Lab>,
    fast_dist: bool
)
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

        let closest_color = take_closest_color(&mut pallete_raw, &mut pallete, color, fast_dist);

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
    let mut fast_dist = false;

    let mut max_size: Option<u32> = None;
    let mut pallete_path: Option<String> = None;

    let mut input_path = String::new();

    {
        let mut parser = ArgumentParser::new();

        parser.refer(&mut fast_dist)
            .add_option(&["-f", "--fast-dist"], StoreTrue,
                "use faster but less accurate distance function"
            );

        parser.refer(&mut max_size)
            .add_option(&["-s", "--size"], StoreOption, "max pallete image size");

        parser.refer(&mut pallete_path)
            .add_option(&["-p", "--pallete"], StoreOption, "pallete image path");

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

    let pallete_image = pallete_path.map(|pallete_path|
    {
        let mut image = image::open(pallete_path).unwrap_or_else(|err|
        {
            cry_about_it(&format!("pallete image error {err:?}"))
        });

        if let Some(max_size) = max_size
        {
            image = image.resize(max_size, max_size, FilterType::Lanczos3);
        }

        image
    });

    let (new_width, new_height) = if let Some(ref pallete_image) = pallete_image
    {
        let (width, height) = (pallete_image.width(), pallete_image.height());

        let (lower, higher) = (width.min(height), width.max(height));

        if image.width() < image.height()
        {
            (lower, higher)
        } else
        {
            (higher, lower)
        }
    } else
    {
        if image.width() < image.height()
        {
            (128, 256)
        } else
        {
            (256, 128)
        }
    };

    let mut image = image.resize_exact(new_width, new_height, FilterType::Lanczos3).into_rgb8();

    let (pallete_raw, pallete) = if let Some(pallete_image) = pallete_image
    {
        pallete_image.into_rgb8().pixels().map(|pixel|
        {
            let color = Color::new(pixel[0], pixel[1], pixel[2]);
            (color, Lab::from(color))
        }).unzip()
    } else
    {
        const MAX_COLOR: usize = 256 / 8;
        const COLORS_AMOUNT: usize = MAX_COLOR.pow(3);

        (0..COLORS_AMOUNT).map(|index|
        {
            let r = (index / MAX_COLOR.pow(2)) * 8;
            let g = ((index / MAX_COLOR) % MAX_COLOR) * 8;
            let b = (index % MAX_COLOR) * 8;

            let color = Color::new(r as u8, g as u8, b as u8);
            (color, Lab::from(color))
        }).unzip()
    };

    palletify(&mut image, pallete_raw, pallete, fast_dist);

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

    #[test]
    fn distance()
    {
        //auto generated by the cringe department
        let tests = [
            (
                Lab{l: 50.0000, a: 2.6772, b: -79.7751},
                Lab{l: 50.0000, a: 0.0000, b: -82.7485},
                2.0425
            ),
            (
                Lab{l: 50.0000, a: 3.1571, b: -77.2803},
                Lab{l: 50.0000, a: 0.0000, b: -82.7485},
                2.8615
            ),
            (
                Lab{l: 50.0000, a: 2.8361, b: -74.0200},
                Lab{l: 50.0000, a: 0.0000, b: -82.7485},
                3.4412
            ),
            (
                Lab{l: 50.0000, a: -1.3802, b: -84.2814},
                Lab{l: 50.0000, a: 0.0000, b: -82.7485},
                1.0000
            ),
            (
                Lab{l: 50.0000, a: -1.1848, b: -84.8006},
                Lab{l: 50.0000, a: 0.0000, b: -82.7485},
                1.0000
            ),
            (
                Lab{l: 50.0000, a: -0.9009, b: -85.5211},
                Lab{l: 50.0000, a: 0.0000, b: -82.7485},
                1.0000
            ),
            (
                Lab{l: 50.0000, a: 0.0000, b: 0.0000},
                Lab{l: 50.0000, a: -1.0000, b: 2.0000},
                2.3669
            ),
            (
                Lab{l: 50.0000, a: -1.0000, b: 2.0000},
                Lab{l: 50.0000, a: 0.0000, b: 0.0000},
                2.3669
            ),
            (
                Lab{l: 50.0000, a: 2.4900, b: -0.0010},
                Lab{l: 50.0000, a: -2.4900, b: 0.0009},
                7.1792
            ),
            (
                Lab{l: 50.0000, a: 2.4900, b: -0.0010},
                Lab{l: 50.0000, a: -2.4900, b: 0.0010},
                7.1792
            ),
            (
                Lab{l: 50.0000, a: 2.4900, b: -0.0010},
                Lab{l: 50.0000, a: -2.4900, b: 0.0011},
                7.2195
            ),
            (
                Lab{l: 50.0000, a: 2.4900, b: -0.0010},
                Lab{l: 50.0000, a: -2.4900, b: 0.0012},
                7.2195
            ),
            (
                Lab{l: 50.0000, a: -0.0010, b: 2.4900},
                Lab{l: 50.0000, a: 0.0009, b: -2.4900},
                4.8045
            ),
            (
                Lab{l: 50.0000, a: -0.0010, b: 2.4900},
                Lab{l: 50.0000, a: 0.0010, b: -2.4900},
                4.8045
            ),
            (
                Lab{l: 50.0000, a: -0.0010, b: 2.4900},
                Lab{l: 50.0000, a: 0.0011, b: -2.4900},
                4.7461
            ),
            (
                Lab{l: 50.0000, a: 2.5000, b: 0.0000},
                Lab{l: 50.0000, a: 0.0000, b: -2.5000},
                4.3065
            ),
            (
                Lab{l: 50.0000, a: 2.5000, b: 0.0000},
                Lab{l: 73.0000, a: 25.0000, b: -18.0000},
                27.1492
            ),
            (
                Lab{l: 50.0000, a: 2.5000, b: 0.0000},
                Lab{l: 61.0000, a: -5.0000, b: 29.0000},
                22.8977
            ),
            (
                Lab{l: 50.0000, a: 2.5000, b: 0.0000},
                Lab{l: 56.0000, a: -27.0000, b: -3.0000},
                31.9030
            ),
            (
                Lab{l: 50.0000, a: 2.5000, b: 0.0000},
                Lab{l: 58.0000, a: 24.0000, b: 15.0000},
                19.4535
            ),
            (
                Lab{l: 50.0000, a: 2.5000, b: 0.0000},
                Lab{l: 50.0000, a: 3.1736, b: 0.5854},
                1.0000
            ),
            (
                Lab{l: 50.0000, a: 2.5000, b: 0.0000},
                Lab{l: 50.0000, a: 3.2972, b: 0.0000},
                1.0000
            ),
            (
                Lab{l: 50.0000, a: 2.5000, b: 0.0000},
                Lab{l: 50.0000, a: 1.8634, b: 0.5757},
                1.0000
            ),
            (
                Lab{l: 50.0000, a: 2.5000, b: 0.0000},
                Lab{l: 50.0000, a: 3.2592, b: 0.3350},
                1.0000
            ),
            (
                Lab{l: 60.2574, a: -34.0099, b: 36.2677},
                Lab{l: 60.4626, a: -34.1751, b: 39.4387},
                1.2644
            ),
            (
                Lab{l: 63.0109, a: -31.0961, b: -5.8663},
                Lab{l: 62.8187, a: -29.7946, b: -4.0864},
                1.2630
            ),
            (
                Lab{l: 61.2901, a: 3.7196, b: -5.3901},
                Lab{l: 61.4292, a: 2.2480, b: -4.9620},
                1.8731
            ),
            (
                Lab{l: 35.0831, a: -44.1164, b: 3.7933},
                Lab{l: 35.0232, a: -40.0716, b: 1.5901},
                1.8645
            ),
            (
                Lab{l: 22.7233, a: 20.0904, b: -46.6940},
                Lab{l: 23.0331, a: 14.9730, b: -42.5619},
                2.0373
            ),
            (
                Lab{l: 36.4612, a: 47.8580, b: 18.3852},
                Lab{l: 36.2715, a: 50.5065, b: 21.2231},
                1.4146
            ),
            (
                Lab{l: 90.8027, a: -2.0831, b: 1.4410},
                Lab{l: 91.1528, a: -1.6435, b: 0.0447},
                1.4441
            ),
            (
                Lab{l: 90.9257, a: -0.5406, b: -0.9208},
                Lab{l: 88.6381, a: -0.8985, b: -0.7239},
                1.5381
            ),
            (
                Lab{l: 6.7747, a: -0.2908, b: -2.4247},
                Lab{l: 5.8714, a: -0.0985, b: -2.2286},
                0.6377
            ),
            (
                Lab{l: 2.0776, a: 0.0795, b: -1.1350},
                Lab{l: 0.9033, a: -0.0636, b: -0.5514},
                0.9082
            ),
        ];

        for test in tests
        {
            let distance = test.0.distance(test.1);

            assert!(close_enough(distance, test.2));
        }
    }
}