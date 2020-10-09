(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("aae21bbf-5b76-45fd-9a91-59a5bdd75c46");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'aae21bbf-5b76-45fd-9a91-59a5bdd75c46' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"bbdbf386-fc57-4809-9f62-b8446056dd46":{"roots":{"references":[{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26816","type":"BoxAnnotation"},{"attributes":{},"id":"26942","type":"Selection"},{"attributes":{"toolbar":{"id":"26948"},"toolbar_location":"above"},"id":"26949","type":"ToolbarBox"},{"attributes":{},"id":"26808","type":"ResetTool"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26918"},"ticker":{"id":"26885"}},"id":"26804","type":"LinearAxis"},{"attributes":{"callback":null},"id":"26849","type":"HoverTool"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26862","type":"VBar"},{"attributes":{"toolbars":[{"id":"26818"},{"id":"26852"}],"tools":[{"id":"26808"},{"id":"26809"},{"id":"26810"},{"id":"26811"},{"id":"26812"},{"id":"26813"},{"id":"26814"},{"id":"26815"},{"id":"26842"},{"id":"26843"},{"id":"26844"},{"id":"26845"},{"id":"26846"},{"id":"26847"},{"id":"26848"},{"id":"26849"}]},"id":"26948","type":"ProxyToolbar"},{"attributes":{"callback":null},"id":"26815","type":"HoverTool"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26908","type":"VBar"},{"attributes":{},"id":"26925","type":"Selection"},{"attributes":{"text":"tau"},"id":"26887","type":"Title"},{"attributes":{},"id":"26918","type":"BasicTickFormatter"},{"attributes":{},"id":"26940","type":"Selection"},{"attributes":{},"id":"26934","type":"BasicTickFormatter"},{"attributes":{},"id":"26848","type":"SaveTool"},{"attributes":{"overlay":{"id":"26817"}},"id":"26812","type":"LassoSelectTool"},{"attributes":{},"id":"26847","type":"UndoTool"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAA8D8OdNpApw30PxSuR+F6FPY/1AY6baDT9T8c6LSBThv4PxdLfrHkF/c/1QY6baDT9T+V/GLJL5b2P1jyiyW/WPc/43oUrkfh+T8ehetRuB75PxdLfrHkF/c/mJmZmZmZ9z8YrkfhehT2P1RVVVVVVfY/lfxiyS+W9j/gehSuR+H5P5iZmZmZmfc/kl8s+cWS9T8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26925"},"selection_policy":{"id":"26926"}},"id":"26867","type":"ColumnDataSource"},{"attributes":{"line_dash":[6],"location":3.4166666666666665},"id":"26884","type":"Span"},{"attributes":{},"id":"26933","type":"BasicTickFormatter"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26869","type":"VBar"},{"attributes":{},"id":"26927","type":"Selection"},{"attributes":{},"id":"26809","type":"PanTool"},{"attributes":{"source":{"id":"26861"}},"id":"26865","type":"CDSView"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26850","type":"BoxAnnotation"},{"attributes":{"data_source":{"id":"26879"},"glyph":{"id":"26880"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26881"},"selection_glyph":null,"view":{"id":"26883"}},"id":"26882","type":"GlyphRenderer"},{"attributes":{},"id":"26832","type":"LinearScale"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26890","type":"VBar"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26919"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26801"}},"id":"26800","type":"LinearAxis"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26891","type":"VBar"},{"attributes":{"axis":{"id":"26800"},"ticker":null},"id":"26803","type":"Grid"},{"attributes":{},"id":"26798","type":"LinearScale"},{"attributes":{"ticks":[0,1,2,3]},"id":"26913","type":"FixedTicker"},{"attributes":{"ticks":[0,1,2,3]},"id":"26885","type":"FixedTicker"},{"attributes":{"data":{"top":{"__ndarray__":"6YVe6IVe4D9nZmZmZmbeP2dmZmZmZu4/WWqlVmql7D/eyI3cyI3YP7vQC73QC9U/uBM7sRM73T+vEzuxEzvdPyZ2Yid2Ytc/lxu5kRu52T8ZuZEbuZHfP5AbuZEbudk/QS/0Qi/04D8LwQ/8wA/cP5AbuZEbudk/q9RKrdRK4z9BL/RCL/TgPyZ2Yid2Ytc/USu1Uiu1wj8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26938"},"selection_policy":{"id":"26939"}},"id":"26889","type":"ColumnDataSource"},{"attributes":{},"id":"26944","type":"Selection"},{"attributes":{"source":{"id":"26879"}},"id":"26883","type":"CDSView"},{"attributes":{},"id":"26930","type":"UnionRenderers"},{"attributes":{"data":{"top":{"__ndarray__":"MzMzMzOzDUC4HoXrUTgPQDCW/GLJrwxA0GkDnTbQDEBtoNMGOu0LQOi0gU4baApAzszMzMzMC0DrUbgehWsLQClcj8L1qApAqA102kAnCkBH4XoUrkcKQMaSXyz5xQlA6LSBThtoCkAqXI/C9agKQClcj8L1qApACtejcD0KC0AGOm2g0wYKQMkvlvxiyQpAaQOdNtDpCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26929"},"selection_policy":{"id":"26930"}},"id":"26879","type":"ColumnDataSource"},{"attributes":{},"id":"26929","type":"Selection"},{"attributes":{"line_dash":[6],"location":2.4166666666666665},"id":"26878","type":"Span"},{"attributes":{},"id":"26830","type":"LinearScale"},{"attributes":{},"id":"26941","type":"UnionRenderers"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26851","type":"PolyAnnotation"},{"attributes":{"below":[{"id":"26800"}],"center":[{"id":"26803"},{"id":"26807"},{"id":"26866"},{"id":"26872"},{"id":"26878"},{"id":"26884"}],"left":[{"id":"26804"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26864"},{"id":"26870"},{"id":"26876"},{"id":"26882"}],"title":{"id":"26887"},"toolbar":{"id":"26818"},"toolbar_location":null,"x_range":{"id":"26792"},"x_scale":{"id":"26796"},"y_range":{"id":"26794"},"y_scale":{"id":"26798"}},"id":"26791","subtype":"Figure","type":"Plot"},{"attributes":{"overlay":{"id":"26816"}},"id":"26810","type":"BoxZoomTool"},{"attributes":{"data_source":{"id":"26873"},"glyph":{"id":"26874"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26875"},"selection_glyph":null,"view":{"id":"26877"}},"id":"26876","type":"GlyphRenderer"},{"attributes":{"source":{"id":"26873"}},"id":"26877","type":"CDSView"},{"attributes":{},"id":"26813","type":"UndoTool"},{"attributes":{},"id":"26938","type":"Selection"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26881","type":"VBar"},{"attributes":{"line_dash":[6],"location":1.4166666666666665},"id":"26872","type":"Span"},{"attributes":{},"id":"26801","type":"BasicTicker"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAAAECkcD0K1yMBQCa/WPKLpQFA6LSBThtoAkBqA5020OkCQCz5xZJfrANA8O7u7u5uBEDrUbgehWsDQOtRuB6FawNAC9ejcD0KA0BKfrHkF0sDQA102kCnDQRAqqqqqqoqA0BQG+i0gU4EQC+W/GLJrwRAThvotIFOBECuR+F6FC4EQE4b6LSBTgRAcD0K16PwBEA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26927"},"selection_policy":{"id":"26928"}},"id":"26873","type":"ColumnDataSource"},{"attributes":{},"id":"26814","type":"SaveTool"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26874","type":"VBar"},{"attributes":{"data_source":{"id":"26867"},"glyph":{"id":"26868"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26869"},"selection_glyph":null,"view":{"id":"26871"}},"id":"26870","type":"GlyphRenderer"},{"attributes":{"children":[{"id":"26949"},{"id":"26947"}]},"id":"26950","type":"Column"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26875","type":"VBar"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26933"},"ticker":{"id":"26913"}},"id":"26838","type":"LinearAxis"},{"attributes":{"data_source":{"id":"26889"},"glyph":{"id":"26890"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26891"},"selection_glyph":null,"view":{"id":"26893"}},"id":"26892","type":"GlyphRenderer"},{"attributes":{},"id":"26928","type":"UnionRenderers"},{"attributes":{"line_dash":[6],"location":0.48076923076923067},"id":"26894","type":"Span"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26903","type":"VBar"},{"attributes":{"source":{"id":"26867"}},"id":"26871","type":"CDSView"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26902","type":"VBar"},{"attributes":{"data_source":{"id":"26895"},"glyph":{"id":"26896"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26897"},"selection_glyph":null,"view":{"id":"26899"}},"id":"26898","type":"GlyphRenderer"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26880","type":"VBar"},{"attributes":{},"id":"26792","type":"DataRange1d"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26897","type":"VBar"},{"attributes":{"source":{"id":"26895"}},"id":"26899","type":"CDSView"},{"attributes":{"source":{"id":"26907"}},"id":"26911","type":"CDSView"},{"attributes":{},"id":"26945","type":"UnionRenderers"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26868","type":"VBar"},{"attributes":{},"id":"26845","type":"WheelZoomTool"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26909","type":"VBar"},{"attributes":{},"id":"26926","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"26907"},"glyph":{"id":"26908"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26909"},"selection_glyph":null,"view":{"id":"26911"}},"id":"26910","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"26804"},"dimension":1,"ticker":null},"id":"26807","type":"Grid"},{"attributes":{},"id":"26811","type":"WheelZoomTool"},{"attributes":{"line_dash":[6],"location":0.41666666666666663},"id":"26866","type":"Span"},{"attributes":{"source":{"id":"26889"}},"id":"26893","type":"CDSView"},{"attributes":{"line_dash":[6],"location":3.480769230769231},"id":"26912","type":"Span"},{"attributes":{},"id":"26939","type":"UnionRenderers"},{"attributes":{},"id":"26796","type":"LinearScale"},{"attributes":{"children":[[{"id":"26791"},0,0],[{"id":"26827"},0,1]]},"id":"26947","type":"GridBox"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26842"},{"id":"26843"},{"id":"26844"},{"id":"26845"},{"id":"26846"},{"id":"26847"},{"id":"26848"},{"id":"26849"}]},"id":"26852","type":"Toolbar"},{"attributes":{"data_source":{"id":"26901"},"glyph":{"id":"26902"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26903"},"selection_glyph":null,"view":{"id":"26905"}},"id":"26904","type":"GlyphRenderer"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26934"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26835"}},"id":"26834","type":"LinearAxis"},{"attributes":{},"id":"26923","type":"Selection"},{"attributes":{"below":[{"id":"26834"}],"center":[{"id":"26837"},{"id":"26841"},{"id":"26894"},{"id":"26900"},{"id":"26906"},{"id":"26912"}],"left":[{"id":"26838"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26892"},{"id":"26898"},{"id":"26904"},{"id":"26910"}],"title":{"id":"26915"},"toolbar":{"id":"26852"},"toolbar_location":null,"x_range":{"id":"26792"},"x_scale":{"id":"26830"},"y_range":{"id":"26794"},"y_scale":{"id":"26832"}},"id":"26827","subtype":"Figure","type":"Plot"},{"attributes":{"line_dash":[6],"location":2.480769230769231},"id":"26906","type":"Span"},{"attributes":{"data":{"top":{"__ndarray__":"ZWZmZmZm7j9OG+i0gU7XP2cDnTbQad8/WfKLJb9Y2j9Bpw102kDTP17JL5b8Yt0/PW2g0wY60T9U8oslv1jaP1ws+cWSX9w/WlVVVVVV2T9SVVVVVVXZPzTQaQOdNuA/ZgOdNtBp3z9m8oslv1jaP0h+seQXS9Y/SH6x5BdL1j84baDTBjrRPz+nDXTaQNM/SH6x5BdL1j8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26923"},"selection_policy":{"id":"26924"}},"id":"26861","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"26850"}},"id":"26844","type":"BoxZoomTool"},{"attributes":{"source":{"id":"26901"}},"id":"26905","type":"CDSView"},{"attributes":{"overlay":{"id":"26851"}},"id":"26846","type":"LassoSelectTool"},{"attributes":{"line_dash":[6],"location":1.4807692307692308},"id":"26900","type":"Span"},{"attributes":{"axis":{"id":"26838"},"dimension":1,"ticker":null},"id":"26841","type":"Grid"},{"attributes":{"data":{"top":{"__ndarray__":"4Qd+4Af+BUDVSq3USq0EQBh6oRd6oQJAGHqhF3qhAkAUO7ETOzECQCu1Uiu10gRAd2IndmKnA0DFTuzETuwCQHIjN3IjNwNAJDdyIzfyA0Bu5EZu5MYCQB/4gR/4gQNAxU7sxE7sAkDTC73QCz0EQNALvdALPQRA0Au90As9BEB6oRd6oRcEQIIf+IEf+ARAhl7ohV5oBUA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26942"},"selection_policy":{"id":"26943"}},"id":"26901","type":"ColumnDataSource"},{"attributes":{},"id":"26919","type":"BasicTickFormatter"},{"attributes":{"data":{"top":{"__ndarray__":"EPzAD/zACUAg+IEf+IELQMEP/MAPfApAdmIndmKnC0A4ciM3ciMOQIZe6IVeaA1Ah17ohV5oDUDYiZ3YiR0NQD7wAz/wAw9Ae6EXeqEXDEAbuZEbuRELQHZiJ3ZipwtAeqEXeqEXDEB0IzdyIzcLQBu5kRu5EQtAFDuxEzsxCkByIzdyIzcLQBu5kRu5EQtAxU7sxE7sCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26944"},"selection_policy":{"id":"26945"}},"id":"26907","type":"ColumnDataSource"},{"attributes":{},"id":"26943","type":"UnionRenderers"},{"attributes":{},"id":"26843","type":"PanTool"},{"attributes":{},"id":"26924","type":"UnionRenderers"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26808"},{"id":"26809"},{"id":"26810"},{"id":"26811"},{"id":"26812"},{"id":"26813"},{"id":"26814"},{"id":"26815"}]},"id":"26818","type":"Toolbar"},{"attributes":{},"id":"26794","type":"DataRange1d"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26817","type":"PolyAnnotation"},{"attributes":{"axis":{"id":"26834"},"ticker":null},"id":"26837","type":"Grid"},{"attributes":{"data_source":{"id":"26861"},"glyph":{"id":"26862"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26863"},"selection_glyph":null,"view":{"id":"26865"}},"id":"26864","type":"GlyphRenderer"},{"attributes":{"text":"mu"},"id":"26915","type":"Title"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26896","type":"VBar"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26863","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"P/ADP/AD9z+SG7mRG7n2Py/0Qi/0QvU/eqEXeqEX9D9IbuRGbuT3P4If+IEf+PQ/MPRCL/RC9T+ZmZmZmZn3Pyd2Yid2YvQ/9kIv9EIv+D+4kRu5kRv7P7ATO7ETO/o/oBd6oRd6+D+mF3qhF3r4P1ZqpVZqpfk/9EIv9EIv+D9GbuRGbuT3P07sxE7sxPg/wA/8wA/8+z8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26940"},"selection_policy":{"id":"26941"}},"id":"26895","type":"ColumnDataSource"},{"attributes":{},"id":"26842","type":"ResetTool"},{"attributes":{},"id":"26835","type":"BasicTicker"}],"root_ids":["26950"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"bbdbf386-fc57-4809-9f62-b8446056dd46","root_ids":["26950"],"roots":{"26950":"aae21bbf-5b76-45fd-9a91-59a5bdd75c46"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();