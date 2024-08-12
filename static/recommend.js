// Replace 'YOUR_API_KEY' below with your API key retrieved from https://www.themoviedb.org
var myAPI = 'a01db429ba34675b395e935cae21a21e';  // global string to be consistent with future usages elsewhere

$(function() {
    $('#movie_list').css('display', 'none');

    $('#autoComplete').blur(function() {
        $('#movie_list').css('display', 'none');
    });

    const source = document.getElementById('autoComplete');
    const inputHandler = function(e) {
        $('#movie_list').css('display', 'block');
        if (e.target.value === "") {
            $('.movie-button').attr('disabled', true);
        } else {
            $('.movie-button').attr('disabled', false);
        }
    }
    source.addEventListener('input', inputHandler);

    $('.fa-arrow-up').click(function(){
        $('html, body').animate({scrollTop: 0}, 'slow');
    });

    $('.app-title').click(function(){
        window.location.href = '/';
    });

    $('.movie-button').on('click', function() {
        let my_api_key = myAPI;
        let title = $('.movie').val();
        $('#movie_list').css('display', 'none');
        if (title === "") {
            $('.results').css('display', 'none');
            $('.fail').css('display', 'block');
        } else {
            load_details(my_api_key, title, true);
        }
    });
});

// Will be invoked when clicking on the recommended movie cards
function recommendcard(id) {
    $("#loader").fadeIn();
    var my_api_key = myAPI;
    load_details(my_api_key, id, false);
}

// Get the details of the movie from the API (based on the name of the movie)
function load_details(my_api_key, search, isQuerySearch) {
    let url;
    if (isQuerySearch) {
        url = 'https://api.themoviedb.org/3/search/movie?api_key=' + my_api_key + '&query=' + search;
    } else {
        url = 'https://api.themoviedb.org/3/movie/' + search + '?api_key=' + my_api_key;
    }
    $.ajax({
        type: 'GET',
        url: url,
        async: false,
        success: function(movie) {
            if (!isQuerySearch) {
                $("#loader").fadeIn();
                $('.fail').css('display', 'none');
                $('.results').delay(1000).css('display', 'block');
                let movie_id = movie.id;
                let movie_title = movie.title;
                let movie_title_org = movie.original_title;
                get_movie_details(movie_id, my_api_key, movie_title, movie_title_org);
            } else if (movie.results.length < 1) {
                $('.fail').css('display', 'block');
                $('.results').css('display', 'none');
                $("#loader").delay(500).fadeOut();
            } else if (movie.results.length == 1) {
                $("#loader").fadeIn();
                $('.fail').css('display', 'none');
                $('.results').delay(1000).css('display', 'block');
                let movie_id = movie.results[0].id;
                let movie_title = movie.results[0].title;
                let movie_title_org = movie.results[0].original_title;
                get_movie_details(movie_id, my_api_key, movie_title, movie_title_org);
            } else {
                $("#loader").fadeIn();
                $('.fail').css('display', 'none');
                $('.results').delay(1000).css('display', 'block');
                details = {
                    'movies_list': movie.results
                };

                $.ajax({
                    type: 'POST',
                    data: JSON.stringify(details),
                    contentType: 'application/json',
                    beforeSend: function() {
                        $("#loader").fadeIn();
                    },
                    url: "/populate-matches",
                    dataType: 'html',
                    complete: function() {
                        $("#loader").delay(1000).fadeOut();
                    },
                    success: function(response) {
                        $('.results').delay(2000).html(response);
                        $('#autoComplete').val('');
                        $('.footer').css('position', 'relative');
                        $('.social').css('padding-bottom', '15px');
                        $('.social').css('margin-bottom', '0px');
                        $(window).scrollTop(0);
                    }
                });
            }
        },
        error: function(error) {
            alert('Invalid Request - ' + error);
            $("#loader").delay(500).fadeOut();
        },
    });
}

// Get all the details of the movie using the movie ID.
function get_movie_details(movie_id, my_api_key, movie_title, movie_title_org) {
    $.ajax({
        type: 'GET',
        url: 'https://api.themoviedb.org/3/movie/' + movie_id + '?api_key=' + my_api_key,
        success: function(movie_details) {
            show_details(movie_details, movie_title, my_api_key, movie_id, movie_title_org);
        },
        error: function(error) {
            alert("API Error! - " + error);
            $("#loader").delay(500).fadeOut();
        },
    });
}

// Passing all the details to Python's Flask for displaying and scraping the movie reviews using IMDb ID
function show_details(movie_details, movie_title, my_api_key, movie_id, movie_title_org) {
    let imdb_id = movie_details.imdb_id;
    let poster = movie_details.poster_path ? 'https://image.tmdb.org/t/p/original' + movie_details.poster_path : 'static/default.jpg';
    let overview = movie_details.overview;
    let genres = movie_details.genres;
    let rating = movie_details.vote_average;
    let vote_count = movie_details.vote_count;
    let release_date = movie_details.release_date;
    let runtime = parseInt(movie_details.runtime);
    let status = movie_details.status;
    let genre_list = genres.map(genre => genre.name);
    let my_genre = genre_list.join(", ");
    runtime = runtime % 60 === 0 ? Math.floor(runtime / 60) + " hour(s)" : Math.floor(runtime / 60) + " hour(s) " + (runtime % 60) + " min(s)";

    // Get the movie cast details
    let movie_cast = get_movie_cast(movie_id, my_api_key);

    // Log cast_ids to debug
    console.log('Cast IDs:', movie_cast.cast_ids);

    // Get individual cast details
    let ind_cast = get_individual_cast(movie_cast.cast_ids, my_api_key);

    // Log ind_cast to debug
    console.log('Individual Cast:', ind_cast);

    // Get recommendations
    let recommendations = get_recommendations(movie_id, my_api_key);

    let details = {
        'title': movie_title,
        'cast_ids': JSON.stringify(movie_cast.cast_ids),
        'cast_names': JSON.stringify(movie_cast.cast_names),
        'cast_chars': JSON.stringify(movie_cast.cast_chars),
        'cast_profiles': JSON.stringify(movie_cast.cast_profiles),
        'cast_bdays': JSON.stringify(ind_cast.cast_bdays),
        'cast_bios': JSON.stringify(ind_cast.cast_bios),
        'cast_places': JSON.stringify(ind_cast.cast_places),
        'imdb_id': imdb_id,
        'poster': poster,
        'genres': my_genre,
        'overview': overview,
        'rating': rating,
        'vote_count': vote_count.toLocaleString(),
        'rel_date': release_date,
        'release_date': new Date(release_date).toDateString().split(' ').slice(1).join(' '),
        'runtime': runtime,
        'status': status,
        'rec_movies': JSON.stringify(recommendations.rec_movies),
        'rec_posters': JSON.stringify(recommendations.rec_posters),
        'rec_movies_org': JSON.stringify(recommendations.rec_movies_org),
        'rec_year': JSON.stringify(recommendations.rec_year),
        'rec_vote': JSON.stringify(recommendations.rec_vote),
        'rec_ids': JSON.stringify(recommendations.rec_ids)
    };

    $.ajax({
        type: 'POST',
        data: details,
        url: "/recommend",
        dataType: 'html',
        complete: function() {
            $("#loader").delay(500).fadeOut();
        },
        success: function(response) {
            $('.results').html(response);
            $('#autoComplete').val('');
            $('.footer').css('position', 'absolute');
            if ($('.movie-content')) {
                $('.movie-content').after('<div class="gototop"><i title="Go to Top" class="fa fa-arrow-up"></i></div>');
            }
            $(window).scrollTop(0);
        }
    });
}

// Getting the details of the cast for the requested movie
function get_movie_cast(movie_id, my_api_key) {
    let cast_ids = [];
    let cast_names = [];
    let cast_chars = [];
    let cast_profiles = [];
    $.ajax({
        type: 'GET',
        url: "https://api.themoviedb.org/3/movie/" + movie_id + "/credits?api_key=" + my_api_key,
        async: false,
        success: function(my_movie) {
            let top_cast = my_movie.cast.length >= 10 ? Array.from({ length: 10 }, (_, i) => i) : Array.from({ length: my_movie.cast.length }, (_, i) => i);
            for (let i of top_cast) {
                cast_ids.push(my_movie.cast[i].id);
                cast_names.push(my_movie.cast[i].name);
                cast_chars.push(my_movie.cast[i].character);
                cast_profiles.push(my_movie.cast[i].profile_path ? "https://image.tmdb.org/t/p/original" + my_movie.cast[i].profile_path : "static/default.jpg");
            }
        },
        error: function(error) {
            alert("API Error! - " + error);
        },
    });

    return {
        'cast_ids': cast_ids,
        'cast_names': cast_names,
        'cast_chars': cast_chars,
        'cast_profiles': cast_profiles
    };
}

// Getting the details of the cast from the cast IDs
function get_individual_cast(cast_ids, my_api_key) {
    let cast_bdays = [];
    let cast_bios = [];
    let cast_places = [];

    // Ensure cast_ids is an array
    if (!Array.isArray(cast_ids)) {
        console.error('Expected cast_ids to be an array, but got:', cast_ids);
        return {
            'cast_bdays': cast_bdays,
            'cast_bios': cast_bios,
            'cast_places': cast_places
        };
    }

    cast_ids.forEach(function(id) {
        $.ajax({
            type: 'GET',
            url: 'https://api.themoviedb.org/3/person/' + id + '?api_key=' + my_api_key,
            async: false,
            success: function(cast_info) {
                cast_bdays.push(cast_info.birthday);
                cast_bios.push(cast_info.biography);
                cast_places.push(cast_info.place_of_birth);
            },
            error: function(error) {
                alert("API Error! - " + error);
            },
        });
    });

    return {
        'cast_bdays': cast_bdays,
        'cast_bios': cast_bios,
        'cast_places': cast_places
    };
}

// Getting the recommendations based on movie ID
function get_recommendations(movie_id, my_api_key) {
    let rec_movies = [];
    let rec_posters = [];
    let rec_movies_org = [];
    let rec_year = [];
    let rec_vote = [];
    let rec_ids = [];
    $.ajax({
        type: 'GET',
        url: 'https://api.themoviedb.org/3/movie/' + movie_id + '/recommendations?api_key=' + my_api_key,
        async: false,
        success: function(recommendations) {
            let rec_list = recommendations.results;
            for (let i of rec_list) {
                rec_movies.push(i.title);
                rec_posters.push(i.poster_path ? 'https://image.tmdb.org/t/p/original' + i.poster_path : 'static/default.jpg');
                rec_movies_org.push(i.original_title);
                rec_year.push(i.release_date.split('-')[0]);
                rec_vote.push(i.vote_average);
                rec_ids.push(i.id);
            }
        },
        error: function(error) {
            alert("API Error! - " + error);
        }
    });

    return {
        'rec_movies': rec_movies,
        'rec_posters': rec_posters,
        'rec_movies_org': rec_movies_org,
        'rec_year': rec_year,
        'rec_vote': rec_vote,
        'rec_ids': rec_ids
    };
}
